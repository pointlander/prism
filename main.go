package main

import (
	"flag"
	"fmt"
	"image/color"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
)

type Mode int

const (
	ModeNone Mode = iota
	ModeOrthogonality
	ModeParallel
	ModeMixed
	ModeEntropy
	ModeVariance
)

func (m Mode) String() string {
	switch m {
	case ModeNone:
		return "none"
	case ModeOrthogonality:
		return "orthogonality"
	case ModeMixed:
		return "mixed"
	case ModeParallel:
		return "parallel"
	case ModeEntropy:
		return "entropy"
	case ModeVariance:
		return "variance"
	}
	return "unknown"
}

const (
	Width  = 16
	Width2 = 16
	Width3 = 16
	Eta    = .6
)

var (
	datum iris.Datum
	once  sync.Once
)

func load() {
	var err error
	datum, err = iris.Load()
	if err != nil {
		panic(err)
	}
	max := 0.0
	for _, item := range datum.Fisher {
		for _, measure := range item.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	for _, item := range datum.Fisher {
		for i, measure := range item.Measures {
			item.Measures[i] = measure / max
		}
	}

	max = 0.0
	for _, item := range datum.Bezdek {
		for _, measure := range item.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	for _, item := range datum.Bezdek {
		for i, measure := range item.Measures {
			item.Measures[i] = measure / max
		}
	}
}

var colors = [...]color.RGBA{
	{R: 0xff, G: 0x00, B: 0x00, A: 255},
	{R: 0x00, G: 0xff, B: 0x00, A: 255},
	{R: 0x00, G: 0x00, B: 0xff, A: 255},
}

func plotData(data *mat.Dense, name string, training []iris.Iris) {
	rows, cols := data.Dims()

	var pc stat.PC
	ok := pc.PrincipalComponents(data, nil)
	if !ok {
		return
	}

	k := 2
	var projection mat.Dense
	projection.Mul(data, pc.VectorsTo(nil).Slice(0, cols, 0, k))

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "iris"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
	p.Legend.Top = true

	for i := 0; i < 3; i++ {
		label := ""
		points := make(plotter.XYs, 0, rows)
		for j := 0; j < rows; j++ {
			if iris.Labels[training[j].Label] != i {
				continue
			}
			label = training[j].Label
			points = append(points, plotter.XY{X: projection.At(j, 0), Y: projection.At(j, 1)})
		}

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Color = colors[i]
		p.Add(scatter)
		p.Legend.Add(fmt.Sprintf("%s", label), scatter)
	}

	err = p.Save(8*vg.Inch, 8*vg.Inch, name)
	if err != nil {
		panic(err)
	}
}

func printTable(out io.Writer, headers []string, rows [][]string) {
	sizes := make([]int, len(headers))
	for i, header := range headers {
		sizes[i] = len(header)
	}
	for _, row := range rows {
		for j, item := range row {
			if length := len(item); length > sizes[j] {
				sizes[j] = length
			}
		}
	}

	fmt.Fprintf(out, "| ")
	for i, header := range headers {
		fmt.Fprintf(out, "%s", header)
		spaces := sizes[i] - len(header)
		for spaces > 0 {
			fmt.Fprintf(out, " ")
			spaces--
		}
		fmt.Fprintf(out, " | ")
	}
	fmt.Fprintf(out, "\n| ")
	for i, header := range headers {
		dashes := len(header)
		if sizes[i] > dashes {
			dashes = sizes[i]
		}
		for dashes > 0 {
			fmt.Fprintf(out, "-")
			dashes--
		}
		fmt.Fprintf(out, " | ")
	}
	fmt.Fprintf(out, "\n")
	for _, row := range rows {
		fmt.Fprintf(out, "| ")
		for i, entry := range row {
			spaces := sizes[i] - len(entry)
			fmt.Fprintf(out, "%s", entry)
			for spaces > 0 {
				fmt.Fprintf(out, " ")
				spaces--
			}
			fmt.Fprintf(out, " | ")
		}
		fmt.Fprintf(out, "\n")
	}
}

type Reduction struct {
	Row, Column int
	Pivot       float64
	Max         float64
	Left, Right *Reduction
}

func (r *Reduction) String() string {
	var serialize func(r *Reduction, label, depth uint) string
	serialize = func(r *Reduction, label, depth uint) string {
		spaces := ""
		for i := uint(0); i < depth; i++ {
			spaces += " "
		}
		left, right := "", ""
		if r.Left != nil {
			left = serialize(r.Left, label, depth+1)
		}
		if r.Right != nil {
			right = serialize(r.Right, label|(1<<depth), depth+1)
		}
		layer := fmt.Sprintf("%s// variance reduction: %f\n", spaces, r.Max)
		layer += fmt.Sprintf("%sif output[%d] > %f {\n", spaces, r.Column, r.Pivot)
		if right == "" {
			layer += fmt.Sprintf("%s label := %d\n", spaces, label|(1<<depth))
		} else {
			layer += fmt.Sprintf("%s\n", right)
		}
		layer += fmt.Sprintf("%s} else {\n", spaces)
		if left == "" {
			layer += fmt.Sprintf("%s label := %d\n", spaces, label)
		} else {
			layer += fmt.Sprintf("%s\n", left)
		}
		layer += fmt.Sprintf("%s}", spaces)
		return layer
	}
	return serialize(r, 0, 0)
}

func (r *Reduction) Label(label, depth uint, cutoff float64, data []float64) uint {
	if r == nil {
		return label
	} else if r.Max < cutoff {
		return label
	} else if data[r.Column] > r.Pivot {
		return r.Right.Label(label|(1<<depth), depth+1, cutoff, data)
	}
	return r.Left.Label(label, depth+1, cutoff, data)
}

func variance(column int, rows [][]float64) float64 {
	n, sum := float64(len(rows)), 0.0
	for _, row := range rows {
		sum += row[column]
	}
	average, variance := sum/n, 0.0
	for _, row := range rows {
		v := row[column] - average
		variance += v * v
	}
	return variance / n
}

func varianceReduction(columns int, rows [][]float64, depth int) *Reduction {
	if len(rows) == 0 {
		return nil
	}

	reduction := Reduction{}
	for k := 0; k < columns; k++ {
		sort.Slice(rows, func(i, j int) bool {
			return rows[i][k] < rows[j][k]
		})
		total := variance(k, rows)
		for i, row := range rows[:len(rows)-1] {
			a, b := variance(k, rows[:i+1]), variance(k, rows[i+1:])
			if cost := total - (a + b); cost > reduction.Max {
				reduction.Max, reduction.Row, reduction.Column, reduction.Pivot = cost, i, k, row[k]
			}
		}
	}
	depth--
	if depth <= 0 {
		return &reduction
	}

	rowscp := make([][]float64, len(rows))
	copy(rowscp, rows)
	sort.Slice(rowscp, func(i, j int) bool {
		return rowscp[i][reduction.Column] < rowscp[j][reduction.Column]
	})
	reduction.Left, reduction.Right =
		varianceReduction(columns, rowscp[:reduction.Row+1], depth),
		varianceReduction(columns, rowscp[reduction.Row+1:], depth)
	return &reduction
}

type Network struct {
	*rand.Rand
	BatchSize     int
	Input, Output tf32.V
	W, B          [4]tf32.V
	Parameters    []*tf32.V
	Ones          tf32.V
	L             [4]tf32.Meta
	Cost          tf32.Meta
}

func NewNetwork(batchSize int) *Network {
	n := Network{
		Rand:       rand.New(rand.NewSource(1)),
		BatchSize:  batchSize,
		Parameters: make([]*tf32.V, 0, 8),
	}
	n.Input, n.Output = tf32.NewV(4, batchSize), tf32.NewV(4, batchSize)
	n.W[0], n.B[0] = tf32.NewV(4, Width), tf32.NewV(Width)
	n.W[1], n.B[1] = tf32.NewV(Width, Width2), tf32.NewV(Width2)
	n.W[2], n.B[2] = tf32.NewV(Width2, Width3), tf32.NewV(Width3)
	n.W[3], n.B[3] = tf32.NewV(Width3, 4), tf32.NewV(4)
	for i := range n.W {
		n.Parameters = append(n.Parameters, &n.W[i], &n.B[i])
	}
	for _, p := range n.Parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, n.Random32(-1, 1))
		}
	}

	n.Ones = tf32.NewV(batchSize)
	for i := 0; i < cap(n.Ones.X); i++ {
		n.Ones.X = append(n.Ones.X, 1)
	}
	last := n.Input.Meta()
	for i := range n.L {
		n.L[i] = tf32.Sigmoid(tf32.Add(tf32.Mul(n.W[i].Meta(), last), n.B[i].Meta()))
		last = n.L[i]
	}
	n.Cost = tf32.Avg(tf32.Sub(n.Ones.Meta(), tf32.Similarity(last, n.Output.Meta())))
	//cost := tf32.Avg(tf32.Quadratic(last, output.Meta()))
	return &n
}

func (n *Network) Random32(a, b float32) float32 {
	return (b-a)*n.Float32() + a
}

func (n *Network) Train(training []iris.Iris, iterations int) plotter.XYs {
	length := len(training)
	data := make([]*iris.Iris, 0, length)
	for i := range training {
		data = append(data, &training[i])
	}

	points := make(plotter.XYs, 0, iterations)
	for i := 0; i < iterations; i++ {
		for i := range data {
			j := i + n.Intn(length-i)
			data[i], data[j] = data[j], data[i]
		}
		total := float32(0.0)
		for j := 0; j < length; j += n.BatchSize {
			for _, p := range n.Parameters {
				p.Zero()
			}
			n.Input.Zero()
			n.Output.Zero()
			n.Ones.Zero()

			values := make([]float32, 0, 4*n.BatchSize)
			for k := 0; k < n.BatchSize; k++ {
				index := (j + k) % length
				for _, measure := range data[index].Measures {
					values = append(values, float32(measure))
				}
			}
			n.Input.Set(values)
			n.Output.Set(values)
			total += tf32.Gradient(n.Cost).X[0]

			norm := float32(0)
			for k, p := range n.Parameters {
				for l, d := range p.D {
					if math.IsNaN(float64(d)) {
						fmt.Println(d, k, l)
						return points
					} else if math.IsInf(float64(d), 0) {
						fmt.Println(d, k, l)
						return points
					}
					norm += d * d
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
			if norm > 1 {
				scaling := 1 / norm
				for _, p := range n.Parameters {
					for l, d := range p.D {
						p.X[l] -= Eta * d * scaling
					}
				}
			} else {
				for _, p := range n.Parameters {
					for l, d := range p.D {
						p.X[l] -= Eta * d
					}
				}
			}
		}
		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
	}
	return points
}

type Result struct {
	Mode       Mode
	Missed     uint
	Mislabeled uint
}

func neuralNetwork(training []iris.Iris, batchSize int, mode Mode) (result Result) {
	result.Mode = mode
	fmt.Println(mode.String())

	network := NewNetwork(batchSize)

	length := len(training)
	learn := func(mode Mode, makePlot bool) {
		iterations := 1000
		switch mode {
		case ModeNone:
		case ModeOrthogonality:
			iterations = 1000
		case ModeParallel:
			iterations = 1000
		case ModeMixed:
			iterations = 1000
		case ModeEntropy:
			iterations = 1000
		case ModeVariance:
			iterations = 10000
		}
		points := network.Train(training, iterations)

		p, err := plot.New()
		if err != nil {
			panic(err)
		}

		p.Title.Text = mode.String()
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		if makePlot {
			err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("epochs_%s.png", mode.String()))
			if err != nil {
				panic(err)
			}
		}
	}

	switch mode {
	case ModeNone:
		learn(ModeNone, true)
	case ModeOrthogonality:
		learn(ModeNone, false)
		weight := tf32.NewV(1)
		weight.X = append(weight.X, 1)
		network.Cost = tf32.Add(network.Cost, tf32.Hadamard(weight.Meta(), tf32.Avg(tf32.Abs(tf32.Orthogonality(network.L[1])))))
		learn(mode, true)
	case ModeParallel:
		learn(ModeNone, false)
		ones := tf32.NewV(((batchSize - 1) * batchSize) / 2)
		for i := 0; i < cap(ones.X); i++ {
			ones.X = append(ones.X, 1)
		}
		weight := tf32.NewV(1)
		weight.X = append(weight.X, 1)
		network.Cost = tf32.Add(network.Cost, tf32.Hadamard(weight.Meta(), tf32.Avg(tf32.Sub(ones.Meta(), tf32.Orthogonality(network.L[1])))))
		learn(mode, true)
	case ModeMixed:
		learn(ModeNone, false)
		pairs := make([]int, 0, length)
		for i, a := range training {
			max, match := -1.0, 0
			for j, b := range training {
				if j == i {
					continue
				}
				sumAB, sumAA, sumBB := 0.0, 0.0, 0.0
				for k, aa := range a.Measures {
					bb := b.Measures[k]
					sumAB += aa * bb
					sumAA += aa * aa
					sumBB += bb * bb
				}
				similarity := sumAB / (math.Sqrt(sumAA) * math.Sqrt(sumBB))
				if similarity > max {
					max, match = similarity, j
				}
			}
			pairs = append(pairs, match)
		}
		mask := tf32.NewV(((batchSize - 1) * batchSize) / 2)
		for i := 0; i < batchSize; i++ {
			for j := i + 1; j < batchSize; j++ {
				if pairs[i] == j || pairs[j] == i {
					mask.X = append(mask.X, 1)
				} else {
					mask.X = append(mask.X, 0)
				}
			}
		}
		weight := tf32.NewV(1)
		weight.X = append(weight.X, 1)
		network.Cost = tf32.Add(network.Cost, tf32.Hadamard(weight.Meta(), tf32.Avg(tf32.Abs(tf32.Sub(mask.Meta(), tf32.Orthogonality(network.L[1]))))))
		learn(mode, true)
	case ModeEntropy:
		learn(ModeNone, false)
		weight := tf32.NewV(1)
		weight.X = append(weight.X, .5)
		network.Cost = tf32.Add(network.Cost, tf32.Hadamard(weight.Meta(), tf32.Avg(tf32.Entropy(tf32.Softmax(tf32.T(network.L[1]))))))
		learn(mode, true)
	case ModeVariance:
		learn(ModeNone, false)
		one := tf32.NewV(1)
		one.X = append(one.X, 1)
		weight := tf32.NewV(1)
		weight.X = append(weight.X, 1)
		network.Cost = tf32.Add(network.Cost, tf32.Hadamard(weight.Meta(), tf32.Sub(one.Meta(), tf32.Avg(tf32.Variance(tf32.T(network.L[1]))))))
		learn(mode, true)
	}

	input := tf32.NewV(4)
	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(network.W[0].Meta(), input.Meta()), network.B[0].Meta()))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(network.W[1].Meta(), l1), network.B[1].Meta()))
	tf32.Static.InferenceOnly = true
	defer func() {
		tf32.Static.InferenceOnly = false
	}()
	points := make([]float64, 0, Width2*length)
	for i := range training {
		values := make([]float32, 0, 4)
		for _, measure := range training[i].Measures {
			values = append(values, float32(measure))
		}
		input.Set(values)
		l2(func(a *tf32.V) {
			for _, value := range a.X {
				points = append(points, float64(value))
			}
		})
	}

	out, err := os.Create(fmt.Sprintf("result_%s.md", mode.String()))
	if err != nil {
		panic(err)
	}
	defer out.Close()

	fmt.Fprintf(out, "# Training cost vs epochs\n")
	fmt.Fprintf(out, "![epochs of %s](epochs_%s.png?raw=true)]\n\n", mode.String(), mode.String())

	depth := 2
	switch mode {
	case ModeNone:
		depth = 2
	case ModeOrthogonality:
		depth = 2
	case ModeParallel:
		depth = 2
	case ModeMixed:
		depth = 2
	case ModeEntropy:
		depth = 2
	case ModeVariance:
		depth = 1
	}
	items := make([][]float64, 0, length)
	for i := 0; i < len(points); i += Width2 {
		item := make([]float64, 0, Width2)
		for j := 0; j < Width2; j++ {
			item = append(item, points[i+j])
		}
		items = append(items, item)
	}
	reduction := varianceReduction(Width2, items, depth)
	fmt.Fprintf(out, "# Decision tree\n")
	fmt.Fprintf(out, "```go\n")
	fmt.Fprintf(out, "%s\n", reduction.String())
	fmt.Fprintf(out, "```\n\n")

	headers, rows := make([]string, 0, Width2+2), make([][]string, 0, length)
	headers = append(headers, "label", "cluster")
	for i := 0; i < Width2; i++ {
		headers = append(headers, fmt.Sprintf("%d", i))
	}

	cutoff := 0.0
	switch mode {
	case ModeNone:
		cutoff = 0.0
	case ModeOrthogonality:
		cutoff = 0.01
	case ModeParallel:
		cutoff = 0.0004
	case ModeMixed:
		cutoff = 0.01
	case ModeEntropy:
		cutoff = 0.00005
	case ModeVariance:
		cutoff = 0.0
	}
	index, counts := 0, make(map[string]map[uint]uint)
	for _, item := range items {
		row := make([]string, 0, Width2+2)
		label, predicted := training[index].Label, reduction.Label(0, 0, cutoff, item)
		count, ok := counts[label]
		if !ok {
			count = make(map[uint]uint)
			counts[label] = count
		}
		count[predicted]++
		row = append(row, label, fmt.Sprintf("%d", predicted))
		for _, value := range item {
			row = append(row, fmt.Sprintf("%f", value))
		}
		rows = append(rows, row)
		index++
	}
	type Triple struct {
		Label     string
		Predicted uint
		Count     uint
	}
	triples := make([]Triple, 0, 8)
	for label, count := range counts {
		for predicted, c := range count {
			triples = append(triples, Triple{
				Label:     label,
				Predicted: predicted,
				Count:     c,
			})
		}
	}
	sort.Slice(triples, func(i, j int) bool {
		return triples[i].Count > triples[j].Count
	})
	labels, used := make(map[string]uint), make(map[uint]bool)
	for _, triple := range triples {
		if _, ok := labels[triple.Label]; !ok {
			if !used[triple.Predicted] {
				labels[triple.Label], used[triple.Predicted] = triple.Predicted, true
			}
		}
	}
	index = 0
	for _, item := range items {
		label, predicted := training[index].Label, reduction.Label(0, 0, cutoff, item)
		if l, ok := labels[label]; !ok || l != predicted {
			result.Mislabeled++
		}
		index++
	}
	fmt.Fprintf(out, "# Output of neural network middle layer\n")
	printTable(out, headers, rows)
	fmt.Fprintf(out, "\n")

	plotData(mat.NewDense(length, Width, points), fmt.Sprintf("embedding_%s.png", mode.String()), training)
	fmt.Fprintf(out, "# PCA of network middle layer\n")
	fmt.Fprintf(out, "![embedding of %s](embedding_%s.png?raw=true)]\n", mode.String(), mode.String())

	for i, x := range items {
		max, match := -1.0, 0
		for j, y := range items {
			if j == i {
				continue
			}
			sumAB, sumAA, sumBB := 0.0, 0.0, 0.0
			for k, a := range x {
				b := y[k]
				sumAB += a * b
				sumAA += a * a
				sumBB += b * b
			}
			similarity := sumAB / (math.Sqrt(sumAA) * math.Sqrt(sumBB))
			if similarity > max {
				max, match = similarity, j
			}
		}
		should := iris.Labels[training[i].Label]
		found := iris.Labels[training[match].Label]
		if should != found {
			result.Missed++
		}
	}

	return
}

var (
	all           = flag.Bool("all", false, "all of the modes")
	orthogonality = flag.Bool("orthogonality", false, "orthogonality mode")
	parallel      = flag.Bool("parallel", false, "parallel mode")
	mixed         = flag.Bool("mixed", false, "mixed mode")
	entropy       = flag.Bool("entropy", false, "entropy mode")
	varianceMode  = flag.Bool("variance", false, "variance mode")
)

func main() {
	flag.Parse()

	results := make([]Result, 0)

	once.Do(load)
	training := datum.Fisher
	length := len(training)
	data := make([]float64, 0, length*4)
	for _, item := range training {
		for _, measure := range item.Measures {
			data = append(data, measure)
		}
	}
	plotData(mat.NewDense(length, 4, data), "iris.png", training)

	results = append(results, neuralNetwork(training, 10, ModeNone))
	if *all || *orthogonality {
		results = append(results, neuralNetwork(training, 150, ModeOrthogonality))
	}
	if *all || *parallel {
		results = append(results, neuralNetwork(training, 150, ModeParallel))
	}
	if *all || *mixed {
		results = append(results, neuralNetwork(training, 150, ModeMixed))
	}
	if *all || *entropy {
		results = append(results, neuralNetwork(training, 60, ModeEntropy))
	}
	if *all || *varianceMode {
		results = append(results, neuralNetwork(training, 150, ModeVariance))
	}

	out, err := os.Create("README.md")
	if err != nil {
		panic(err)
	}
	defer out.Close()

	headers, rows := make([]string, 0, Width2+2), make([][]string, 0, len(results))
	headers = append(headers, "mode", "missed", "mislabeled")
	for _, result := range results {
		row := []string{result.Mode.String(), fmt.Sprintf("%d", result.Missed), fmt.Sprintf("%d", result.Mislabeled)}
		rows = append(rows, row)
	}
	printTable(out, headers, rows)
}

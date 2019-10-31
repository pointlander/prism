package main

import (
	"flag"
	"fmt"
	"image/color"
	"io"
	"math"
	"os"
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
	ModeRaw
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
	case ModeRaw:
		return "raw"
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

func plotData(embeddings *Embeddings, name string) {
	length := len(embeddings.Embeddings)
	values := make([]float64, 0, Width2*length)
	for _, embedding := range embeddings.Embeddings {
		values = append(values, embedding.Features...)
	}
	data := mat.NewDense(length, embeddings.Columns, values)
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
			if iris.Labels[embeddings.Embeddings[j].Label] != i {
				continue
			}
			label = embeddings.Embeddings[j].Label
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

type Result struct {
	Mode        Mode
	Reduction   *Reduction
	Consistency uint
	Mislabeled  uint
}

type Context struct {
	Count int
}

func (c *Context) neuralNetwork(d int, label, count uint, embeddings *Embeddings, batchSize int, mode Mode) (result Result) {
	result.Mode = mode
	if d <= 0 {
		return
	}
	fmt.Printf("%s %d\n", mode.String(), count)

	network := NewNetwork(batchSize)

	length := len(embeddings.Embeddings)
	training := make([]iris.Iris, 0, length)
	for _, embedding := range embeddings.Embeddings {
		training = append(training, embedding.Iris)
	}
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
		if count > 0 {
			iterations *= 10
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
			err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("results/epochs_%s_%d.png", mode.String(), c.Count))
			if err != nil {
				panic(err)
			}
			c.Count++
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

	embeddins := network.Embeddings(training)

	depth := 1
	switch mode {
	case ModeNone:
		depth = 1
	case ModeOrthogonality:
		depth = 1
	case ModeParallel:
		depth = 1
	case ModeMixed:
		depth = 1
	case ModeEntropy:
		depth = 1
	case ModeVariance:
		depth = 1
	}
	result.Reduction = embeddins.VarianceReduction(depth, label, count)

	left := result.Reduction.Left
	resultA := c.neuralNetwork(d-1, left.Label, left.Depth, left.Embeddings, len(left.Embeddings.Embeddings), mode)
	if resultA.Reduction != nil {
		result.Reduction.Left = resultA.Reduction
	}

	right := result.Reduction.Right
	resultB := c.neuralNetwork(d-1, right.Label, right.Depth, right.Embeddings, len(right.Embeddings.Embeddings), mode)
	if resultB.Reduction != nil {
		result.Reduction.Right = resultB.Reduction
	}

	return result
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

	err := os.MkdirAll("results", 0700)
	if err != nil {
		panic(err)
	}

	once.Do(load)
	training := datum.Fisher

	results := make([]Result, 0)
	fmt.Println(ModeRaw.String())
	length := len(training)
	embeddings := Embeddings{
		Columns:    4,
		Embeddings: make([]Embedding, 0, length),
	}
	for _, item := range training {
		embedding := Embedding{
			Iris:     item,
			Features: make([]float64, 0, 4),
		}
		for _, measure := range item.Measures {
			embedding.Features = append(embedding.Features, measure)
		}
		embeddings.Embeddings = append(embeddings.Embeddings, embedding)
	}
	reduction := embeddings.VarianceReduction(2, 0, 0)
	out, err := os.Create(fmt.Sprintf("results/result_%s.md", ModeRaw.String()))
	if err != nil {
		panic(err)
	}
	defer out.Close()
	reduction.PrintTable(out, ModeRaw, 0)
	result := Result{
		Mode:        ModeRaw,
		Reduction:   reduction,
		Mislabeled:  reduction.GetMislabeled(0),
		Consistency: reduction.GetConsistency(),
	}
	results = append(results, result)
	add := func(batchSize int, mode Mode) {
		out, err := os.Create(fmt.Sprintf("results/result_%s.md", mode.String()))
		if err != nil {
			panic(err)
		}
		defer out.Close()

		embeddings, context := Embeddings{}, Context{}
		for _, item := range training {
			embedding := Embedding{
				Iris: item,
			}
			embeddings.Embeddings = append(embeddings.Embeddings, embedding)
		}

		result := context.neuralNetwork(2, 0, 0, &embeddings, batchSize, mode)

		cutoff := 0.0
		switch mode {
		case ModeNone:
			cutoff = 0.0006
		case ModeOrthogonality:
			cutoff = 0.01
		case ModeParallel:
			cutoff = 0.0004
		case ModeMixed:
			cutoff = 0.01
		case ModeEntropy:
			cutoff = 0.0
		case ModeVariance:
			cutoff = 0.0
		}
		result.Reduction.PrintTable(out, mode, cutoff)
		result.Mislabeled = result.Reduction.GetMislabeled(cutoff)
		result.Consistency = result.Reduction.GetConsistency()

		results = append(results, result)
	}
	add(10, ModeNone)
	if *all || *orthogonality {
		add(150, ModeOrthogonality)
	}
	if *all || *parallel {
		add(150, ModeParallel)
	}
	if *all || *mixed {
		add(150, ModeMixed)
	}
	if *all || *entropy {
		add(60, ModeEntropy)
	}
	if *all || *varianceMode {
		add(150, ModeVariance)
	}

	readme, err := os.Create("README.md")
	if err != nil {
		panic(err)
	}
	defer readme.Close()

	headers, rows := make([]string, 0, Width2+2), make([][]string, 0, len(results))
	headers = append(headers, "mode", "consistency", "mislabeled")
	for _, result := range results {
		row := []string{result.Mode.String(), fmt.Sprintf("%d", result.Consistency), fmt.Sprintf("%d", result.Mislabeled)}
		rows = append(rows, row)
	}
	printTable(readme, headers, rows)
}

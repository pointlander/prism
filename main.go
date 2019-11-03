// Copyright 2019 The Prism Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"image/color"
	"io"
	"math"
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
	ModeRaw
	ModeOrthogonality
	ModeParallel
	ModeMixed
	ModeEntropy
	ModeVariance
	NumberOfModes
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
	Width  = 4
	Width2 = 4
	Width3 = 4
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
	Seed        int64
	Reduction   *Reduction
	Consistency uint
	Entropy     float64
}

type Context struct {
	Count int
	Seed  int64
}

func (c *Context) neuralNetwork(d int, label, count uint, embeddings *Embeddings, batchSize int, mode Mode) (result Result) {
	result = Result{
		Mode: mode,
		Seed: c.Seed,
	}
	length := len(embeddings.Embeddings)
	if d <= 0 || length == 0 {
		return
	}
	fmt.Printf("%d %s %d\n", c.Seed, mode.String(), count)

	network := NewNetwork(c.Seed, batchSize)
	training := make([]iris.Iris, 0, length)
	for _, embedding := range embeddings.Embeddings {
		training = append(training, embedding.Iris)
	}
	learn := func(mode Mode, makePlot bool) {
		iterations := 1000
		switch mode {
		case ModeNone:
			iterations = 1000
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

		if makePlot && c.Seed == 1 {
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
		orthogonality := tf32.Avg(tf32.Abs(tf32.Orthogonality(network.L[1])))
		network.Cost = tf32.Add(network.Cost, tf32.Hadamard(orthogonality, orthogonality))
		learn(mode, true)
	case ModeParallel:
		ones := tf32.NewV(((batchSize - 1) * batchSize) / 2)
		for i := 0; i < cap(ones.X); i++ {
			ones.X = append(ones.X, 1)
		}
		parallel := tf32.Avg(tf32.Sub(ones.Meta(), tf32.Orthogonality(network.L[1])))
		network.Cost = tf32.Add(network.Cost, tf32.Hadamard(parallel, parallel))
		learn(mode, true)
	case ModeMixed:
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
		mixed := tf32.Avg(tf32.Abs(tf32.Sub(mask.Meta(), tf32.Orthogonality(network.L[1]))))
		network.Cost = tf32.Add(network.Cost, tf32.Hadamard(mixed, mixed))
		learn(mode, true)
	case ModeEntropy:
		entropy := tf32.Avg(tf32.Entropy(tf32.Softmax(tf32.T(network.L[1]))))
		network.Cost = tf32.Add(network.Cost, tf32.Hadamard(entropy, entropy))
		learn(mode, true)
	case ModeVariance:
		one := tf32.NewV(1)
		one.X = append(one.X, 1)
		variance := tf32.Sub(one.Meta(), tf32.Avg(tf32.Variance(tf32.T(network.L[1]))))
		network.Cost = tf32.Add(network.Cost, tf32.Hadamard(variance, variance))
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
	options = struct {
		all           *bool
		orthogonality *bool
		parallel      *bool
		mixed         *bool
		entropy       *bool
		variance      *bool
		experiments   *int64
	}{
		all:           flag.Bool("all", false, "all of the modes"),
		orthogonality: flag.Bool("orthogonality", false, "orthogonality mode"),
		parallel:      flag.Bool("parallel", false, "parallel mode"),
		mixed:         flag.Bool("mixed", false, "mixed mode"),
		entropy:       flag.Bool("entropy", false, "entropy mode"),
		variance:      flag.Bool("variance", false, "variance mode"),
		experiments:   flag.Int64("experiments", 1, "number of experiments"),
	}
)

func main() {
	flag.Parse()

	if *options.all {
		*options.orthogonality = true
		*options.parallel = true
		*options.mixed = true
		*options.entropy = true
		*options.variance = true
	}

	err := os.MkdirAll("results", 0700)
	if err != nil {
		panic(err)
	}

	once.Do(load)
	training := datum.Fisher

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
		Seed:        1,
		Reduction:   reduction,
		Entropy:     reduction.GetEntropy(0),
		Consistency: reduction.GetConsistency(),
	}

	add := func(seed int64, batchSize int, mode Mode) Result {
		embeddings, context := Embeddings{}, Context{Seed: seed}
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
			cutoff = 0.0
		case ModeOrthogonality:
			cutoff = 0.0
		case ModeParallel:
			cutoff = 0.0
		case ModeMixed:
			cutoff = 0.0
		case ModeEntropy:
			cutoff = 0.0
		case ModeVariance:
			cutoff = 0.0
		}
		if seed == 1 {
			out, err := os.Create(fmt.Sprintf("results/result_%s.md", mode.String()))
			if err != nil {
				panic(err)
			}
			defer out.Close()
			result.Reduction.PrintTable(out, mode, cutoff)
		}
		result.Entropy = result.Reduction.GetEntropy(cutoff)
		result.Consistency = result.Reduction.GetConsistency()

		return result
	}

	fini := make(chan []Result, 8)
	experiment := func(seed int64) {
		results := make([]Result, 0, 8)
		results = append(results, result, add(seed, 10, ModeNone))
		if *options.orthogonality {
			results = append(results, add(seed, 150, ModeOrthogonality))
		}
		if *options.parallel {
			results = append(results, add(seed, 150, ModeParallel))
		}
		if *options.mixed {
			results = append(results, add(seed, 150, ModeMixed))
		}
		if *options.entropy {
			results = append(results, add(seed, 60, ModeEntropy))
		}
		if *options.variance {
			results = append(results, add(seed, 150, ModeVariance))
		}
		fini <- results
	}

	for i := int64(0); i < *options.experiments; i++ {
		go experiment(i + 1)
	}
	experiments := make([][]Result, 0, *options.experiments)
	for i := int64(0); i < *options.experiments; i++ {
		experiments = append(experiments, <-fini)
	}

	readme, err := os.Create("README.md")
	if err != nil {
		panic(err)
	}
	defer readme.Close()

	type Statistic struct {
		Mode       Mode
		Sum        float64
		SumSquared float64
	}
	statistics := make([]Statistic, NumberOfModes)
	for i := range statistics {
		statistics[i].Mode = Mode(i)
	}
	for _, experiment := range experiments {
		if experiment[1].Seed == 1 {
			sort.Slice(experiment, func(i, j int) bool {
				return experiment[i].Entropy < experiment[j].Entropy
			})
			headers, rows := make([]string, 0, 3), make([][]string, 0, len(experiment))
			headers = append(headers, "mode", "consistency", "entropy")
			for _, result := range experiment {
				row := []string{result.Mode.String(), fmt.Sprintf("%d", result.Consistency), fmt.Sprintf("%f", result.Entropy)}
				rows = append(rows, row)
			}
			printTable(readme, headers, rows)
		}
		for _, result := range experiment {
			statistics[result.Mode].Sum += result.Entropy
			statistics[result.Mode].SumSquared += result.Entropy * result.Entropy
		}
	}

	sort.Slice(statistics, func(i, j int) bool {
		return statistics[i].Sum < statistics[j].Sum
	})
	n := float64(len(experiments))
	headers, rows := make([]string, 0, 2), make([][]string, 0, len(statistics))
	headers = append(headers, "mode", "entropy mean", "entropy variance")
	for _, statistic := range statistics {
		mean := statistic.Sum / n
		variance := math.Abs(statistic.SumSquared/n - mean*mean)
		row := []string{statistic.Mode.String(), fmt.Sprintf("%f", mean), fmt.Sprintf("%f", variance)}
		rows = append(rows, row)
	}
	fmt.Fprintf(readme, "\n")
	printTable(readme, headers, rows)
}

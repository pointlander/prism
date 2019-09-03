package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
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

const (
	Width     = 16
	Width2    = 16
	Width3    = 16
	BatchSize = 10
	Eta       = .1
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

func plotData(data *mat.Dense, name string) {
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
			if iris.Labels[datum.Fisher[j].Label] != i {
				continue
			}
			label = datum.Fisher[j].Label
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

	missed := 0
	for i := 0; i < rows; i++ {
		max, match := -1.0, 0
		for j := 0; j < rows; j++ {
			if j == i {
				continue
			}
			sumAB, sumAA, sumBB := 0.0, 0.0, 0.0
			for k := 0; k < cols; k++ {
				a, b := data.At(i, k), data.At(j, k)
				sumAB += a * b
				sumAA += a * a
				sumBB += b * b
			}
			similarity := sumAB / (math.Sqrt(sumAA) * math.Sqrt(sumBB))
			if similarity > max {
				max, match = similarity, j
			}
		}
		should := iris.Labels[datum.Fisher[i].Label]
		found := iris.Labels[datum.Fisher[match].Label]
		if should != found {
			fmt.Println(max)
			missed++
		}
	}
	fmt.Println("missed", missed)
}

func neuralNetwork(name string, orthogonality bool) {
	fmt.Println(name)
	rnd := rand.New(rand.NewSource(1))
	random32 := func(a, b float32) float32 {
		return (b-a)*rnd.Float32() + a
	}
	ones := tf32.NewV(BatchSize)
	for i := 0; i < cap(ones.X); i++ {
		ones.X = append(ones.X, 1)
	}
	weight := tf32.NewV(1)
	weight.X = append(weight.X, 1)
	input, output := tf32.NewV(4, BatchSize), tf32.NewV(4, BatchSize)
	w1, b1, w2, b2 := tf32.NewV(4, Width), tf32.NewV(Width), tf32.NewV(Width, Width2), tf32.NewV(Width2)
	w3, b3, w4, b4 := tf32.NewV(Width2, Width3), tf32.NewV(Width3), tf32.NewV(Width3, 4), tf32.NewV(4)
	parameters := []*tf32.V{&w1, &b1, &w2, &b2, &w3, &b3, &w4, &b4}
	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, random32(-1, 1))
		}
	}
	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(w1.Meta(), input.Meta()), b1.Meta()))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
	l3 := tf32.Sigmoid(tf32.Add(tf32.Mul(w3.Meta(), l2), b3.Meta()))
	l4 := tf32.Sigmoid(tf32.Add(tf32.Mul(w4.Meta(), l3), b4.Meta()))
	cost := tf32.Avg(tf32.Sub(ones.Meta(), tf32.Similarity(l4, output.Meta())))
	//cost := tf32.Avg(tf32.Quadratic(l4, output.Meta()))
	if orthogonality {
		cost = tf32.Add(cost, tf32.Hadamard(weight.Meta(), tf32.Avg(tf32.Abs(tf32.Orthogonality(l2)))))
	}

	length := len(datum.Fisher)
	data := make([]*iris.Iris, 0, length)
	for i := range datum.Fisher {
		data = append(data, &datum.Fisher[i])
	}

LEARN:
	for i := 0; i < 1000; i++ {
		for i := range data {
			j := i + rnd.Intn(length-i)
			data[i], data[j] = data[j], data[i]
		}
		total := float32(0.0)
		for j := 0; j < length; j += BatchSize {
			for _, p := range parameters {
				p.Zero()
			}

			values := make([]float32, 0, 4*BatchSize)
			for k := 0; k < BatchSize; k++ {
				index := (j + k) % length
				for _, measure := range data[index].Measures {
					values = append(values, float32(measure))
				}
			}
			input.Set(values)
			output.Set(values)
			total += tf32.Gradient(cost).X[0]

			norm := float32(0)
			for k, p := range parameters {
				for l, d := range p.D {
					if math.IsNaN(float64(d)) {
						fmt.Println(d, k, l)
						break LEARN
					} else if math.IsInf(float64(d), 0) {
						fmt.Println(d, k, l)
						break LEARN
					}
					norm += d * d
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
			if norm > 1 {
				scaling := 1 / norm
				for _, p := range parameters {
					for l, d := range p.D {
						p.X[l] -= Eta * d * scaling
					}
				}
			} else {
				for _, p := range parameters {
					for l, d := range p.D {
						p.X[l] -= Eta * d
					}
				}
			}
		}
		fmt.Println(total)
		if math.IsNaN(float64(total)) {
			break
		}
		if orthogonality {
			if total < 16.56 {
				break
			}
		} else if total < 1 {
			break
		}
	}

	input = tf32.NewV(4)
	l1 = tf32.Sigmoid(tf32.Add(tf32.Mul(w1.Meta(), input.Meta()), b1.Meta()))
	l2 = tf32.Sigmoid(tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta()))
	tf32.Static.InferenceOnly = true
	defer func() {
		tf32.Static.InferenceOnly = false
	}()
	points := make([]float64, 0, Width2*length)
	for i := range datum.Fisher {
		values := make([]float32, 0, 4)
		for _, measure := range datum.Fisher[i].Measures {
			values = append(values, float32(measure))
		}
		input.Set(values)
		l2(func(a *tf32.V) {
			for _, value := range a.X {
				points = append(points, float64(value))
			}
		})
	}
	plotData(mat.NewDense(length, Width, points), name)
}

func main() {
	once.Do(load)

	length := len(datum.Fisher)
	data := make([]float64, 0, length*4)
	for _, item := range datum.Fisher {
		for _, measure := range item.Measures {
			data = append(data, measure)
		}
	}
	plotData(mat.NewDense(length, 4, data), "iris.png")

	neuralNetwork("embedding.png", false)
	neuralNetwork("orthogonality_embedding.png", true)
}

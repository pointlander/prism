package main

import (
	"fmt"
	"math"
	"os"

	"github.com/pointlander/datum/iris"
)

var MaxEntropy = math.Log2(3)

// Embeddings is a set of embeddings
type Embeddings struct {
	Columns    int
	Network    *Network
	Embeddings []Embedding
}

// Embedding is an embedding with a label and features
type Embedding struct {
	iris.Iris
	Source   int
	Features []float64
}

// Copy makes a copy of the embeddings
func (e *Embeddings) Copy() Embeddings {
	embeddings := Embeddings{
		Columns:    e.Columns,
		Embeddings: make([]Embedding, len(e.Embeddings)),
	}
	copy(embeddings.Embeddings, e.Embeddings)
	return embeddings
}

// Variance computes the variance for the features with column
func (e *Embeddings) Variance(column int) float64 {
	n, sum := float64(len(e.Embeddings)), 0.0
	for _, row := range e.Embeddings {
		sum += row.Features[column]
	}
	average, variance := sum/n, 0.0
	for _, row := range e.Embeddings {
		v := row.Features[column] - average
		variance += v * v
	}
	return variance / n
}

// PivotVariance computes the variance for the left and right features with column
func (e *Embeddings) PivotVariance(column int, pivot float64) (left, right float64) {
	nLeft, nRight, sumLeft, sumRight := 0, 0, 0.0, 0.0
	for _, row := range e.Embeddings {
		if value := row.Features[column]; value > pivot {
			nRight++
			sumRight += value
		} else {
			nLeft++
			sumLeft += value
		}
	}
	averageLeft, averageRight := sumLeft, sumRight
	if nLeft != 0 {
		averageLeft /= float64(nLeft)
	}
	if nRight != 0 {
		averageRight /= float64(nRight)
	}
	for _, row := range e.Embeddings {
		if value := row.Features[column]; value > pivot {
			v := value - averageRight
			right += v * v
		} else {
			v := value - averageLeft
			left += v * v
		}
	}
	if nLeft != 0 {
		left /= float64(nLeft)
	}
	if nRight != 0 {
		right /= float64(nRight)
	}
	return left, right
}

// VarianceReduction implements variance reduction algorithm
func (e *Embeddings) VarianceReduction(depth int, label, count uint) *Reduction {
	length := len(e.Embeddings)
	if length == 0 {
		return nil
	}

	reduction := Reduction{
		Embeddings: e,
		Label:      label,
		Depth:      count,
	}
	if depth <= 0 {
		return &reduction
	}

	for k := 0; k < e.Columns; k++ {
		total := e.Variance(k)
		for _, row := range e.Embeddings {
			pivot := row.Features[k]
			a, b := e.PivotVariance(k, pivot)
			if cost := total - (a + b); cost > reduction.Max {
				reduction.Max, reduction.Column, reduction.Pivot = cost, k, pivot
			}
		}
	}

	left := Embeddings{
		Columns:    e.Columns,
		Network:    e.Network,
		Embeddings: make([]Embedding, 0, length),
	}
	right := Embeddings{
		Columns:    e.Columns,
		Network:    e.Network,
		Embeddings: make([]Embedding, 0, length),
	}
	for _, row := range e.Embeddings {
		if row.Features[reduction.Column] > reduction.Pivot {
			right.Embeddings = append(right.Embeddings, row)
		} else {
			left.Embeddings = append(left.Embeddings, row)
		}
	}
	reduction.Left, reduction.Right =
		left.VarianceReduction(depth-1, label, count+1),
		right.VarianceReduction(depth-1, label|(1<<count), count+1)
	return &reduction
}

// PrintTable prints a table of embeddings
func (r *Reduction) PrintTable(out *os.File, mode Mode, cutoff float64) {
	if out == nil {
		return
	}

	fmt.Fprintf(out, "# Training cost vs epochs\n")
	fmt.Fprintf(out, "![epochs of %s](epochs_%s.png?raw=true)]\n\n", mode.String(), mode.String())

	fmt.Fprintf(out, "# Decision tree\n")
	fmt.Fprintf(out, "```go\n")
	fmt.Fprintf(out, "%s\n", r.String())
	fmt.Fprintf(out, "```\n\n")

	headers, rows := make([]string, 0, Width2+2), make([][]string, 0, 256)
	headers = append(headers, "label", "cluster")
	for i := 0; i < r.Embeddings.Columns; i++ {
		headers = append(headers, fmt.Sprintf("%d", i))
	}

	var load func(r *Reduction)
	load = func(r *Reduction) {
		if r == nil {
			return
		}
		if (r.Left == nil && r.Right == nil) || r.Max < cutoff {
			for _, item := range r.Embeddings.Embeddings {
				row := make([]string, 0, r.Embeddings.Columns+2)
				label, predicted := item.Label, r.Label
				row = append(row, label, fmt.Sprintf("%d", predicted))
				for _, value := range item.Features {
					row = append(row, fmt.Sprintf("%f", value))
				}
				rows = append(rows, row)
			}
			return
		}
		load(r.Left)
		load(r.Right)
	}
	load(r.Left)
	load(r.Right)

	fmt.Fprintf(out, "# Output of neural network middle layer\n")
	printTable(out, headers, rows)
	fmt.Fprintf(out, "\n")

	plotData(r.Embeddings, fmt.Sprintf("results/embedding_%s.png", mode.String()))
	fmt.Fprintf(out, "# PCA of network middle layer\n")
	fmt.Fprintf(out, "![embedding of %s](embedding_%s.png?raw=true)]\n", mode.String(), mode.String())
}

// GetEntropy gets the entropy
func (r *Reduction) GetEntropy(cutoff float64) (entropy float64) {
	histograms := make(map[uint][3]uint)
	var count func(r *Reduction)
	count = func(r *Reduction) {
		if r == nil {
			return
		}
		if (r.Left == nil && r.Right == nil) || r.Max < cutoff {
			predicted := r.Label
			for _, item := range r.Embeddings.Embeddings {
				histogram := histograms[predicted]
				histogram[iris.Labels[item.Label]]++
				histograms[predicted] = histogram
			}
			return
		}
		count(r.Left)
		count(r.Right)
	}
	count(r.Left)
	count(r.Right)

	t := uint(0)
	for _, histogram := range histograms {
		for _, counts := range histogram {
			t += counts
		}
	}
	total := float64(t)
	for _, histogram := range histograms {
		s := uint(0)
		for _, counts := range histogram {
			s += counts
		}
		e, sum := 0.0, float64(s)
		for _, c := range histogram {
			if c == 0 {
				continue
			}
			counts := float64(c)
			e += counts * math.Log2(counts) / sum
		}
		entropy += sum * (math.Log2(sum) - e) / total
	}
	return entropy / MaxEntropy
}

// GetConsistency returns zero if the data is self consistent
func (r *Reduction) GetConsistency() (consistency uint) {
	embeddings := r.Embeddings
	for i, x := range embeddings.Embeddings {
		max, match := -1.0, 0
		for j, y := range embeddings.Embeddings {
			if j == i {
				continue
			}
			sumAB, sumAA, sumBB := 0.0, 0.0, 0.0
			for k, a := range x.Features {
				b := y.Features[k]
				sumAB += a * b
				sumAA += a * a
				sumBB += b * b
			}
			similarity := sumAB / (math.Sqrt(sumAA) * math.Sqrt(sumBB))
			if similarity > max {
				max, match = similarity, j
			}
		}
		should := iris.Labels[embeddings.Embeddings[i].Label]
		found := iris.Labels[embeddings.Embeddings[match].Label]
		if should != found {
			consistency++
		}
	}
	return consistency
}

// Reduction is the result of variance reduction
type Reduction struct {
	Embeddings  *Embeddings
	Label       uint
	Depth       uint
	Column      int
	Pivot       float64
	Max         float64
	Left, Right *Reduction
}

// String converts the reduction to a string representation
func (r *Reduction) String() string {
	var serialize func(r *Reduction, depth uint) string
	serialize = func(r *Reduction, depth uint) string {
		spaces := ""
		for i := uint(0); i < depth; i++ {
			spaces += " "
		}
		left, right := "", ""
		var labelLeft, labelRight uint
		if r.Left != nil {
			labelLeft = r.Left.Label
			if r.Left.Left != nil || r.Left.Right != nil {
				left = serialize(r.Left, depth+1)
			}
		}
		if r.Right != nil {
			labelRight = r.Right.Label
			if r.Right.Left != nil || r.Right.Right != nil {
				right = serialize(r.Right, depth+1)
			}
		}
		layer := fmt.Sprintf("%s// variance reduction: %f\n", spaces, r.Max)
		layer += fmt.Sprintf("%sif output[%d] > %f {\n", spaces, r.Column, r.Pivot)
		if right == "" {
			layer += fmt.Sprintf("%s label := %d\n", spaces, labelRight)
		} else {
			layer += fmt.Sprintf("%s\n", right)
		}
		layer += fmt.Sprintf("%s} else {\n", spaces)
		if left == "" {
			layer += fmt.Sprintf("%s label := %d\n", spaces, labelLeft)
		} else {
			layer += fmt.Sprintf("%s\n", left)
		}
		layer += fmt.Sprintf("%s}", spaces)
		return layer
	}
	return serialize(r, 0)
}

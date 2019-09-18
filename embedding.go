package main

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/pointlander/datum/iris"
)

// Embeddings is a set of embeddings
type Embeddings struct {
	Columns    int
	Embeddings []Embedding
}

// Embedding is an embedding with a label and features
type Embedding struct {
	Label    string
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

// VarianceReduction implements variance reduction algorithm
func (e *Embeddings) VarianceReduction(depth int) *Reduction {
	if len(e.Embeddings) == 0 {
		return nil
	}

	reduction := Reduction{}
	for k := 0; k < e.Columns; k++ {
		sort.Slice(e.Embeddings, func(i, j int) bool {
			return e.Embeddings[i].Features[k] < e.Embeddings[j].Features[k]
		})
		total := e.Variance(k)
		for i, row := range e.Embeddings[:len(e.Embeddings)-1] {
			left := Embeddings{
				Columns:    e.Columns,
				Embeddings: e.Embeddings[:i+1],
			}
			right := Embeddings{
				Columns:    e.Columns,
				Embeddings: e.Embeddings[i+1:],
			}
			a, b := left.Variance(k), right.Variance(k)
			if cost := total - (a + b); cost > reduction.Max {
				reduction.Max, reduction.Row, reduction.Column, reduction.Pivot = cost, i, k, row.Features[k]
			}
		}
	}
	depth--
	if depth <= 0 {
		return &reduction
	}

	rowscp := make([]Embedding, len(e.Embeddings))
	copy(rowscp, e.Embeddings)
	sort.Slice(rowscp, func(i, j int) bool {
		return rowscp[i].Features[reduction.Column] < rowscp[j].Features[reduction.Column]
	})
	left := Embeddings{
		Columns:    e.Columns,
		Embeddings: rowscp[:reduction.Row+1],
	}
	right := Embeddings{
		Columns:    e.Columns,
		Embeddings: rowscp[reduction.Row+1:],
	}
	reduction.Left, reduction.Right = left.VarianceReduction(depth), right.VarianceReduction(depth)
	return &reduction
}

// PrintTable prints a table of embeddings
func (e *Embeddings) PrintTable(mode Mode, cutoff float64, reduction *Reduction) {
	out, err := os.Create(fmt.Sprintf("result_%s.md", mode.String()))
	if err != nil {
		panic(err)
	}
	defer out.Close()

	fmt.Fprintf(out, "# Training cost vs epochs\n")
	fmt.Fprintf(out, "![epochs of %s](epochs_%s.png?raw=true)]\n\n", mode.String(), mode.String())

	fmt.Fprintf(out, "# Decision tree\n")
	fmt.Fprintf(out, "```go\n")
	fmt.Fprintf(out, "%s\n", reduction.String())
	fmt.Fprintf(out, "```\n\n")

	headers, rows := make([]string, 0, Width2+2), make([][]string, 0, len(e.Embeddings))
	headers = append(headers, "label", "cluster")
	for i := 0; i < e.Columns; i++ {
		headers = append(headers, fmt.Sprintf("%d", i))
	}

	for _, item := range e.Embeddings {
		row := make([]string, 0, e.Columns+2)
		label, predicted := item.Label, reduction.Label(0, 0, cutoff, item.Features)
		row = append(row, label, fmt.Sprintf("%d", predicted))
		for _, value := range item.Features {
			row = append(row, fmt.Sprintf("%f", value))
		}
		rows = append(rows, row)
	}

	fmt.Fprintf(out, "# Output of neural network middle layer\n")
	printTable(out, headers, rows)
	fmt.Fprintf(out, "\n")

	plotData(e, fmt.Sprintf("embedding_%s.png", mode.String()))
	fmt.Fprintf(out, "# PCA of network middle layer\n")
	fmt.Fprintf(out, "![embedding of %s](embedding_%s.png?raw=true)]\n", mode.String(), mode.String())
}

// GetMislabeled computes how many embeddings are mislabeled
func (e *Embeddings) GetMislabeled(cutoff float64, reduction *Reduction) (mislabeled uint) {
	counts := make(map[string]map[uint]uint)
	for _, item := range e.Embeddings {
		label, predicted := item.Label, reduction.Label(0, 0, cutoff, item.Features)
		count, ok := counts[label]
		if !ok {
			count = make(map[uint]uint)
			counts[label] = count
		}
		count[predicted]++
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
	for _, item := range e.Embeddings {
		label, predicted := item.Label, reduction.Label(0, 0, cutoff, item.Features)
		if l, ok := labels[label]; !ok || l != predicted {
			mislabeled++
		}
	}
	return mislabeled
}

// GetConsistency returns zero if the data is self consistent
func (e *Embeddings) GetConsistency() (consistency uint) {
	for i, x := range e.Embeddings {
		max, match := -1.0, 0
		for j, y := range e.Embeddings {
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
		should := iris.Labels[e.Embeddings[i].Label]
		found := iris.Labels[e.Embeddings[match].Label]
		if should != found {
			consistency++
		}
	}
	return consistency
}

// Reduction is the result of variance reduction
type Reduction struct {
	Row, Column int
	Pivot       float64
	Max         float64
	Left, Right *Reduction
}

// String converts the reduction to a string representation
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

// Label creates a label for some data based on the reduction
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

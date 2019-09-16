package main

import (
	"fmt"
	"sort"
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

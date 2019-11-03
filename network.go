// Copyright 2019 The Prism Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/plot/plotter"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
)

// Network is a neural network
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

// NewNetwork creates a new neural network
func NewNetwork(seed int64, batchSize int) *Network {
	n := Network{
		Rand:       rand.New(rand.NewSource(seed)),
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

// Random32 generates a random number between a and b
func (n *Network) Random32(a, b float32) float32 {
	return (b-a)*n.Float32() + a
}

// Train trains the neural network on training data for iterations
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

// Embeddings generates the embeddings
func (n *Network) Embeddings(training []iris.Iris) Embeddings {
	input := tf32.NewV(4)
	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(n.W[0].Meta(), input.Meta()), n.B[0].Meta()))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(n.W[1].Meta(), l1), n.B[1].Meta()))
	/*tf32.Static.InferenceOnly = true
	defer func() {
		tf32.Static.InferenceOnly = false
	}()*/
	embeddings := Embeddings{
		Columns:    Width2,
		Network:    n,
		Embeddings: make([]Embedding, 0, len(training)),
	}
	for i := range training {
		values := make([]float32, 0, 4)
		for _, measure := range training[i].Measures {
			values = append(values, float32(measure))
		}
		input.Set(values)
		embedding := Embedding{
			Iris:     training[i],
			Source:   i,
			Features: make([]float64, 0, Width2),
		}
		l2(func(a *tf32.V) {
			for _, value := range a.X {
				embedding.Features = append(embedding.Features, float64(value))
			}
		})
		embeddings.Embeddings = append(embeddings.Embeddings, embedding)
	}
	return embeddings
}

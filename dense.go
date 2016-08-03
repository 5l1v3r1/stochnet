package stochnet

import (
	"math"
	"math/rand"
)

const denseLearningDelta = 1e-2

type Dense struct {
	Weights [][]float32
	Biases  []float32
}

func NewDense(inSize, outSize int) *Dense {
	res := &Dense{
		Weights: make([][]float32, outSize),
		Biases:  make([]float32, outSize),
	}
	for i := range res.Weights {
		res.Weights[i] = make([]float32, inSize)
	}
	return res
}

func (d *Dense) Randomize() *Dense {
	if len(d.Weights) == 0 {
		return d
	}
	randScale := 2 / float32(math.Sqrt(float64(len(d.Weights[0]))))
	for _, weights := range d.Weights {
		for i := range weights {
			weights[i] = float32(rand.NormFloat64()) * randScale
		}
	}
	return d
}

func (d *Dense) Apply(in BoolVec) BoolVec {
	outputs := make([]bool, len(d.Weights))
	outSums := make([]float32, len(d.Weights))
	for i, weights := range d.Weights {
		sum := d.Biases[i]
		for j, input := range in.Activations() {
			if input {
				sum += weights[j]
			}
		}
		outSums[i] = sum
		if float32(rand.NormFloat64()) < sum {
			outputs[i] = true
		}
	}
	return &denseResult{
		Dense:   d,
		Input:   in,
		Output:  outputs,
		OutSums: outSums,
	}
}

type denseResult struct {
	Dense   *Dense
	Input   BoolVec
	Output  []bool
	OutSums []float32
}

func (d *denseResult) Activations() []bool {
	return d.Output
}

func (d *denseResult) Learn(changeFlags []bool) {
	delta := float32(denseLearningDelta)
	inPosDesirability := make([]float32, len(d.Input.Activations()))
	for i, change := range changeFlags {
		deriv := float32(math.Exp(float64(-d.OutSums[i] * d.OutSums[i])))
		if d.Output[i] == change {
			d.Dense.Biases[i] -= delta * deriv
			for j, in := range d.Input.Activations() {
				if in {
					d.Dense.Weights[i][j] -= delta * deriv
				}
				inPosDesirability[j] -= d.Dense.Weights[i][j] * deriv
			}
		} else {
			d.Dense.Biases[i] += delta * deriv
			for j, in := range d.Input.Activations() {
				if in {
					d.Dense.Weights[i][j] += delta * deriv
				}
				inPosDesirability[j] += d.Dense.Weights[i][j] * deriv
			}
		}
	}

	upstream := make([]bool, len(inPosDesirability))
	for i, x := range inPosDesirability {
		upstream[i] = ((x > 0) != d.Input.Activations()[i])
	}
	d.Input.Learn(upstream)
}

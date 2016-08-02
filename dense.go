package stochnet

import "math/rand"

const denseLearningDelta = 1

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

func (d *Dense) Apply(in BoolVec) BoolVec {
	outputs := make([]bool, len(d.Weights))
	for i, weights := range d.Weights {
		sum := d.Biases[i]
		for j, input := range in.Activations() {
			if input {
				sum += weights[j]
			}
		}
		if float32(rand.NormFloat64()) < sum {
			outputs[i] = true
		}
	}
	return &denseResult{
		Dense:  d,
		Input:  in,
		Output: outputs,
	}
}

type denseResult struct {
	Dense  *Dense
	Input  BoolVec
	Output []bool
}

func (d *denseResult) Activations() []bool {
	return d.Output
}

func (d *denseResult) Learn(changeFlags []bool) {
	delta := float32(denseLearningDelta)
	inPosDesirability := make([]float32, len(d.Input.Activations()))
	for i, change := range changeFlags {
		if !change {
			continue
		}
		if d.Output[i] {
			d.Dense.Biases[i] -= delta
			for j, in := range d.Input.Activations() {
				if in {
					d.Dense.Weights[i][j] -= delta
				}
				inPosDesirability[j] -= d.Dense.Weights[i][j]
			}
		} else {
			d.Dense.Biases[i] += delta
			for j, in := range d.Input.Activations() {
				if in {
					d.Dense.Weights[i][j] += delta
				}
				inPosDesirability[j] += d.Dense.Weights[i][j]
			}
		}
	}

	upstream := make([]bool, len(inPosDesirability))
	for i, x := range inPosDesirability {
		upstream[i] = ((x > 0) != d.Input.Activations()[i])
		if rand.Intn(3) == 0 {
			upstream[i] = false
		}
	}
	d.Input.Learn(upstream)
}

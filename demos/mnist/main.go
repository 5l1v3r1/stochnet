package main

import (
	"log"
	"math/rand"
	"time"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/stochnet"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	network := stochnet.Network{
		stochnet.NewDense(28*28, 1000).Randomize(),
		stochnet.NewDense(1000, 10).Randomize(),
	}

	log.Println("Training ...")

	dataset := mnist.LoadTrainingDataSet()
	for _, item := range dataset.Samples {
		bitmap := intensitiesToBools(item.Intensities)
		output := network.Apply(stochnet.ConstBoolVec(bitmap))
		change := make([]bool, 10)
		for i, o := range output.Activations() {
			if o != (i == item.Label) {
				change[i] = true
			}
		}
		output.Learn(change)
	}

	log.Println("Cross validating ...")

	hist := mnist.LoadTestingDataSet().CorrectnessHistogram(func(data []float64) int {
		boolVec := stochnet.ConstBoolVec(intensitiesToBools(data))
		output := network.Apply(boolVec).Activations()
		var idx int
		for i, o := range output {
			if o {
				idx = i
			}
		}
		return idx
	})
	log.Println("Histogram:", hist)
}

func intensitiesToBools(f []float64) []bool {
	res := make([]bool, len(f))
	for i, x := range f {
		if x > 0.5 {
			res[i] = true
		} else {
			res[i] = false
		}
	}
	return res
}

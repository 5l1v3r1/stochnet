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
		stochnet.NewDense(28*28, 300).Randomize(),
		stochnet.NewDense(300, 200).Randomize(),
		stochnet.NewDense(200, 10).Randomize(),
	}

	log.Println("Training ...")

	var epoch int
	for {
		dataset := mnist.LoadTrainingDataSet()
		perm := rand.Perm(len(dataset.Samples))
		for i := 0; i < 5000; i++ {
			item := dataset.Samples[perm[i]]
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

		log.Printf("Cross validating epoch %d ...", epoch)
		epoch++

		hist := mnist.LoadTestingDataSet().CorrectnessHistogram(func(data []float64) int {
			boolVec := stochnet.ConstBoolVec(intensitiesToBools(data))
			output := network.Apply(boolVec).Activations()
			var idxs []int
			for i, o := range output {
				if o {
					idxs = append(idxs, i)
				}
			}
			if len(idxs) == 0 {
				return rand.Intn(10)
			}
			return idxs[rand.Intn(len(idxs))]
		})
		log.Println("Histogram:", hist)

		for i, layer := range network {
			if i != 0 {
				network[i-1].(*stochnet.Dense).AddOutputs(5)
				layer.(*stochnet.Dense).AddInputs(5)
			}
		}
	}
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

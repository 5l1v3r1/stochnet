package main

import (
	"log"
	"math/rand"
	"time"

	"github.com/unixpickle/stochnet"
)

const testCount = 1000

func main() {
	rand.Seed(time.Now().UnixNano())
	network := stochnet.Network{
		stochnet.NewDense(2, 2),
		stochnet.NewDense(2, 1),
	}

	// The network doesn't learn if there are too many
	// things to learn, hence I setup a perfect network
	// and commented out some values.

	network[0].(*stochnet.Dense).Biases[0] = -20
	network[0].(*stochnet.Dense).Biases[1] = -20
	network[0].(*stochnet.Dense).Weights[0][0] = 40
	network[0].(*stochnet.Dense).Weights[0][1] = -40
	network[0].(*stochnet.Dense).Weights[1][0] = -40
	network[0].(*stochnet.Dense).Weights[1][1] = 40
	network[1].(*stochnet.Dense).Biases[0] = -10
	network[1].(*stochnet.Dense).Weights[0][0] = 20
	//network[1].(*stochnet.Dense).Weights[0][1] = 20

	log.Println("Initial correct rate:", correctRate(network))
	for i := 0; i < 1000000; i++ {
		in1 := rand.Intn(2) == 0
		in2 := rand.Intn(2) == 0
		out := (in1 && !in2) || (!in1 && in2)
		actualOut := network.Apply(stochnet.ConstBoolVec([]bool{in1, in2}))
		actualOut.Learn([]bool{actualOut.Activations()[0] != out})
	}
	log.Println("Final correct rate:", correctRate(network))
}

func correctRate(layer stochnet.Layer) float64 {
	var numCorrect int
	for i := 0; i < testCount; i++ {
		in1 := rand.Intn(2) == 0
		in2 := rand.Intn(2) == 0
		out := (in1 && !in2) || (!in1 && in2)
		actualOut := layer.Apply(stochnet.ConstBoolVec([]bool{in1, in2}))
		if actualOut.Activations()[0] == out {
			numCorrect++
		}
	}
	return float64(numCorrect) / float64(testCount)
}

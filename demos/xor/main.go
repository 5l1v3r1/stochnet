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
		stochnet.NewDense(2, 30).Randomize(),
		stochnet.NewDense(30, 1).Randomize(),
	}

	log.Println("Initial correct rate:", correctRate(network))
	for i := 0; i < 100000; i++ {
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

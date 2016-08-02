// Package stochnet implements an experimental kind of
// neural network that uses randomness in place of
// continuous numerical values.
package stochnet

type BoolVec interface {
	Activations() []bool
	Learn(changeFlags []bool)
}

func ConstBoolVec(activations []bool) BoolVec {
	return constBoolVec(activations)
}

type Layer interface {
	Apply(in BoolVec) BoolVec
}

type Network []Layer

func (n Network) Apply(in BoolVec) BoolVec {
	for _, x := range n {
		in = x.Apply(in)
	}
	return in
}

type constBoolVec []bool

func (c constBoolVec) Activations() []bool {
	return c
}

func (c constBoolVec) Learn(changeFlags []bool) {
}

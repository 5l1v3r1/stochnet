package stochnet

// Concat concatenates boolean vectors to form a new
// boolean vector.
func Concat(vecs ...BoolVec) BoolVec {
	var output []bool
	for _, v := range vecs {
		output = append(output, v.Activations()...)
	}
	return &concatRes{
		Vecs: vecs,
		Out:  output,
	}
}

type concatRes struct {
	Vecs []BoolVec
	Out  []bool
}

func (c *concatRes) Activations() []bool {
	return c.Out
}

func (c *concatRes) Learn(changeFlags []bool) {
	var idx int
	for i, v := range c.Vecs {
		l := len(v.Activations())
		v.Learn(changeFlags[i : i+l])
		idx += l
	}
}

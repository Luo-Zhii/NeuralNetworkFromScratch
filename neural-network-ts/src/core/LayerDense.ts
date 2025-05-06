import * as math from "mathjs";

class LayerDense {
  weights: math.Matrix;
  biases: math.Matrix;
  output: math.Matrix | null = null;

  constructor(nInputs: number, nNeurons: number) {
    // create random weights
    const rawWeights = math.random([nInputs, nNeurons], -1, 1);
    this.weights = math.multiply(rawWeights, 0.01) as math.Matrix;

    // create bisases 
    this.biases = math.zeros([1, nNeurons]) as math.Matrix;
  }

  forward(inputs: math.Matrix) {
    // this.output = inputs.dot(weights) + biases
    this.output = math.add(math.multiply(inputs, this.weights), this.biases);
  }
}

export default LayerDense;

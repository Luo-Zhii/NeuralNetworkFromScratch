import * as math from 'mathjs';
import LayerDense from './core/LayerDense';
import formatMatrix from './utils/formatMatrix';

// create random data
const X = math.matrix(
	Array.from({ length: 5 }, () => [Math.random(), Math.random()]),
) as math.Matrix;

// code
const dense1 = new LayerDense(2, 3);
dense1.forward(X);

console.log(dense1.output);

// display the output
const outputElement = document.getElementById('output');

if (!outputElement) {
	throw new Error('Output element not found');
}

outputElement.textContent = `Output:\n${formatMatrix(dense1.output!)}`;

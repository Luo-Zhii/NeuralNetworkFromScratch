import * as math from "mathjs";
import LayerDense from "./core/LayerDense";


const X = math.matrix(Array.from({ length: 5 }, () =>
  [Math.random(), Math.random()]
)) as math.Matrix;

const dense1 = new LayerDense(2, 3);
dense1.forward(X);


console.log(dense1.output);


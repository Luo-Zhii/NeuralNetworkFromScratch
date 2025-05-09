import * as math from 'mathjs';

const formatMatrix = (matrix: math.Matrix): string => {
	const data = matrix.toArray() as number[][];
	return data.map((row) => row.join(', ')).join('\n');
};

export default formatMatrix;

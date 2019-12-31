const inputDims = [1, 2, 2, 8];
const filterDims = [1, 1, 8, 1];
const outputDims = [1, 2, 2, 1];

const inputValue = 0.01;
const filterValue = 0.01;
const biasValue = 0.1;

const noBias = 0;
const noRelu = false;

function product(shape) {
  let result = 1;
  for (let i = 0; i < shape.length; i++) {
    result = result * shape[i];
  }
  return result;
}

async function createWebNNConv(filterValue, biasValue, hasRelu) {
  const nn = navigator.ml.getNeuralNetworkContext();
  const options = {
    "backend": "WebML",
    "prefer": "sustained"
  };
  const model = await nn.createModel(options);
  let operandIndex = 0;

  const inputDesc = {type: nn.TENSOR_FLOAT32, dimensions: inputDims};
  const filterDesc = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3], filterDims[0], filterDims[1], filterDims[2]]};
  const biasDesc = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3]]};
  const outputDesc = {type: nn.TENSOR_FLOAT32, dimensions: outputDims};
  const intDesc = {type: nn.INT32};

  const input = operandIndex++;
  model.addOperand(inputDesc);
  const filter = operandIndex++;
  model.addOperand(filterDesc);
  const bias = operandIndex++;
  model.addOperand(biasDesc);
  const pad = operandIndex++;
  model.addOperand(intDesc);
  const act = operandIndex++;
  model.addOperand(intDesc);
  const stride = operandIndex++;
  model.addOperand(intDesc);
  const output = operandIndex++;
  model.addOperand(outputDesc);

  const filterData = new Float32Array(product(filterDims));
  filterData.fill(filterValue);
  const biasData = new Float32Array(filterDims[3]);
  biasData.fill(biasValue);
  model.setOperandValue(filter, filterData);
  model.setOperandValue(bias, biasData);
  model.setOperandValue(pad, new Int32Array([(filterDims[1]-1)/2]));
  model.setOperandValue(act, new Int32Array([hasRelu?nn.FUSE_RELU:nn.FUSE_NONE]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [input, filter, bias, pad, pad, pad, pad, stride, stride, act], [output]);

  model.identifyInputsAndOutputs([input], [output]);
  await model.finish();
  return model;
}

async function WebNNConvSharingWebGL(){
  // const inputData = new Float32Array(product(inputDims));
  // inputData.fill(inputValue);
  const inputData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]);
  // const inputData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
  const input = tf.tensor(inputData, inputDims);
  const filterData = new Float32Array(product(filterDims));
  filterData.fill(filterValue);
  const filter = tf.tensor(filterData, filterDims);
  const biasData = new Float32Array(filterDims[3]);
  biasData.fill(biasValue);
  const bias = tf.tensor(biasData, [filterDims[3]]);
  // Conv with tf.js
  let convOutput = tf.conv2d(input, filter, 1, 'same');
  result = await convOutput.data();
  console.log("The result of Conv with TF.js:\n" + result);

  const nn = navigator.ml.getNeuralNetworkContext();
  const model = await createWebNNConv(filterValue, noBias, noRelu);
  const compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  const execution = await compilation.createExecution();

  const inputTensor = input;
  const inputTexture = tf.backend().getTexture(inputTensor.dataId);
  const gl = tf.backend().getGPGPUContext().gl;
  const texShape = tf.backend().getDataInfo(inputTensor.dataId).texShape;
  execution.setInput(0, gl, inputTexture, texShape[0], texShape[1]);
  // const output = await tf.zeros(outputDims).data();
  // execution.setOutput(0, output);
  const outputTensor = tf.tensor([0, 0, 0, 0], outputDims);
  const outputTexture = tf.backend().getTexture(outputTensor.dataId);
  const outputTexShape = tf.backend().getDataInfo(outputTensor.dataId).texShape;
  execution.setOutput(6, gl, outputTexture, outputTexShape[0], outputTexShape[1]);
  await execution.startCompute();
  result = await outputTensor.data();
  console.log("The result of WebNN Conv with sharing WebGL:\n" + result);
}

async function test() {
  document.getElementById('backend').innerText = `TF.js sets backend as WebWebGL`;
  document.getElementById('size').innerText = `conv2d input dims: [${inputDims}] and filter dims: [${filterDims}]`;
  await WebNNConvSharingWebGL();
  document.getElementById('output').innerHTML += `Done <br/>`;
}

async function main() {
  await tf.ready();
  await tf.setBackend('webgl');
  document.getElementById('start').disabled = false;
  document.getElementById('start').addEventListener('click', () => {test();})
}
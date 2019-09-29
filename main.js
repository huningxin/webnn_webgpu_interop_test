const inputDims = [1, 100, 100, 100];
const filterDims = [3, 3, 100, 100];

const inputValue = 0.01;
const filterValue = 0.01;
const biasValue = 0.1;
const bias2Value = 0.2;

const iterations = 100;

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

  // inputDims [n,h,w,i]
  // filterDims [h,w,i,o]
  const inputDesc = {type: nn.TENSOR_FLOAT32, dimensions: inputDims};
  const filterDesc = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3], filterDims[0], filterDims[1], filterDims[2]]};
  const biasDesc = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3]]};
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
  model.addOperand(inputDesc);

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

let webgpu_exe = false;
function executeWebNNForGPU(execution, input, output) {
  if (webgpu_exe) {
    const commandEncoder = tf.backend().device.createCommandEncoder();
    commandEncoder.setNnGraphInput(input, 0, execution);
    commandEncoder.setNnGraphOutput(output, 0, execution);
    commandEncoder.executeNnGraph(execution);
    device.getQueue().submit([commandEncoder.finish()]);
  } else {
    execution.setInputGPUBuffer(0, input);
    execution.setOutputGPUBuffer(0, output);
    execution.startCompute();
  }
}

async function tfConv2d(inputDims, filterDims){
  const inputData = new Float32Array(product(inputDims));
  inputData.fill(inputValue);
  const input = tf.tensor(inputData, inputDims);
  const filterData = new Float32Array(product(filterDims));
  filterData.fill(filterValue);
  const filter = tf.tensor(filterData, filterDims);
  const biasData = new Float32Array(filterDims[3]);
  biasData.fill(biasValue);
  const bias = tf.tensor(biasData, [filterDims[3]]);
  //warm up
  let convOutput = tf.conv2d(input, filter, 1, 'same');
  let addOutput = tf.add(convOutput, bias);
  let reluOutput = tf.relu(addOutput);
  let result = await reluOutput.data();

  const start = performance.now();
  for(let i = 0; i < iterations; i++){
    convOutput = tf.conv2d(input, filter, 1, 'same');
    addOutput = tf.add(convOutput, bias);
    reluOutput = tf.relu(addOutput);
    result = await reluOutput.data();
  }
  const elapsedTime = ((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebGPU conv2d/add/relu elapsed time: ${elapsedTime} ms <br/>`;
  console.log(result);
}

async function WebNNConvGPUWithTf(inputDims, filterDims) {
  const nn = navigator.ml.getNeuralNetworkContext();
  const nnConv = await createWebNNConv(filterValue, 0, false);
  const compilation = await nnConv.createCompilation();
  compilation.setGPUDevice(tf.backend().device);
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  const execution = await compilation.createExecution();

  const inputData = new Float32Array(product(inputDims));
  inputData.fill(inputValue);
  const input = tf.tensor(inputData, inputDims);
  const biasData = new Float32Array(filterDims[3]);
  biasData.fill(biasValue);
  const bias = tf.tensor(biasData, [filterDims[3]]);
  const output = tf.zeros(inputDims);

  // warm up
  const inputBuffer = tf.backend().getBuffer(input.dataId);
  const outputBuffer = tf.backend().getBuffer(output.dataId);
  executeWebNNForGPU(execution, inputBuffer, outputBuffer);
  let addOutput = tf.add(output, bias);
  let reluOutput = tf.relu(addOutput);
  let result = await reluOutput.data();

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    executeWebNNForGPU(execution, inputBuffer, outputBuffer);
    addOutput = tf.add(output, bias);
    reluOutput = tf.relu(addOutput);
    result = await reluOutput.data();
  }
  const elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv2d interops with WebGPU add/relu via WebGPUBuffer elapsed time: ${elapsedTime} ms <br/>`;
  console.log(result);
}

async function WebNNConvGPU(inputDims, filterDims) {
  const nn = navigator.ml.getNeuralNetworkContext();
  const nnConv = await createWebNNConv(filterValue, biasValue, true);
  const compilation = await nnConv.createCompilation();
  compilation.setGPUDevice(tf.backend().device);
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  const execution = await compilation.createExecution();

  const inputData = new Float32Array(product(inputDims));
  inputData.fill(inputValue);
  const inputTensor = tf.tensor(inputData, inputDims);
  const outputTensor = tf.zeros(inputDims);
  const inputBuffer = tf.backend().getBuffer(inputTensor.dataId);
  const outputBuffer = tf.backend().getBuffer(outputTensor.dataId);

  // warm up
  executeWebNNForGPU(execution, inputBuffer, outputBuffer);
  // bypass convertAndCacheOnCPU
  let info = tf.backend().tensorMap.get(outputTensor.dataId);
  let data = await tf.backend().getBufferData(info);
  let result = new Float32Array(data);

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    executeWebNNForGPU(execution, inputBuffer, outputBuffer);
    // bypass convertAndCacheOnCPU
    info = tf.backend().tensorMap.get(outputTensor.dataId);
    data = await tf.backend().getBufferData(info);
    result = new Float32Array(data);
  }
  const elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv2d/bias/relu via WebGPUBuffer elapsed time: ${elapsedTime} ms <br/>`;
  console.log(result);  
}

async function WebNNConvCPU(inputDims, filterDims) {
  const nn = navigator.ml.getNeuralNetworkContext();
  const model = await createWebNNConv(filterValue, biasValue, true);
  const compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  const execution = await compilation.createExecution();

  const inputData = new Float32Array(product(inputDims));
  inputData.fill(inputValue);
  const input = await tf.tensor(inputData, inputDims).data();
  const output = await tf.zeros(inputDims).data();

  execution.setInput(0, input);
  execution.setOutput(0, output);

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    await execution.startCompute();
  }
  const elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv2d/bias/relu via ArrayBuffer elapsed time: ${elapsedTime} ms <br/>`;
  console.log(output);
}

async function WebNNConvCPUWithTf(inputDims, filterDims) {
  const nn = navigator.ml.getNeuralNetworkContext();
  const model = await createWebNNConv(filterValue, 0, false);
  const compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  const execution = await compilation.createExecution();

  const inputData = new Float32Array(product(inputDims));
  inputData.fill(inputValue);
  const input = await tf.tensor(inputData, inputDims).data();
  const output = await tf.zeros(inputDims).data();
  const biasData = new Float32Array(filterDims[3]);
  biasData.fill(biasValue);
  const biasTensor = tf.tensor(biasData, [filterDims[3]]);

  execution.setInput(0, input);
  execution.setOutput(0, output);
  await execution.startCompute();

  let outputTensor = tf.tensor(output, inputDims);
  let addOutput = tf.add(outputTensor, biasTensor);
  let reluOutput = tf.relu(addOutput);
  let result = await reluOutput.data();
  let start = performance.now();
  for (let i = 0; i < iterations; i++) {
    await execution.startCompute();
    outputTensor = tf.tensor(output, inputDims);
    addOutput = tf.add(outputTensor, biasTensor);
    reluOutput = tf.relu(addOutput);
    result = await reluOutput.data();
  }
  const elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv2d interops with WebGPU add/relu via ArrayBuffer elapsed time: ${elapsedTime} ms <br/>`;
  console.log(result);
}

async function tfConv2dx2(inputDims, filterDims){
  const inputData = new Float32Array(product(inputDims));
  inputData.fill(inputValue);
  const input = tf.tensor(inputData, inputDims);
  const filterData = new Float32Array(product(filterDims));
  filterData.fill(filterValue);
  const filter = tf.tensor(filterData, filterDims);
  const biasData = new Float32Array(filterDims[3]);
  biasData.fill(biasValue);
  const bias = tf.tensor(biasData, [filterDims[3]]);
  const bias2Data = new Float32Array(filterDims[3]);
  bias2Data.fill(bias2Value);
  const bias2 = tf.tensor(bias2Data, [filterDims[3]]);
  //warm up
  let im0 = tf.conv2d(input, filter, 1, 'same');
  let im1 = tf.add(im0, bias);
  let im2 = tf.relu(im1);
  let im3 = tf.conv2d(im2, filter, 1, 'same');
  let im4 = tf.add(im3, bias2);
  let im5 = tf.relu(im4);
  let result = await im5.data();
  const start = performance.now();
  for(let i = 0; i < iterations; i++){
    im0 = tf.conv2d(input, filter, 1, 'same');
    im1 = tf.add(im0, bias);
    im2 = tf.relu(im1);
    im3 = tf.conv2d(im2, filter, 1, 'same');
    im4 = tf.add(im3, bias2);
    im5 = tf.relu(im4);
    result = await im5.data();
  }
  console.log(result);
  const elapsedTime = ((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebGPU conv2d/add/relu x2 via WebGPUBuffer elapsed time: ${elapsedTime} ms <br/>`;
}

async function WebNNConvGPUx2(inputDims, filterDims) {
  const nn = navigator.ml.getNeuralNetworkContext();
  const options = {
    "backend": "WebML",
    "prefer": "sustained"
  };
  let model = await nn.createModel(options);
  let operandIndex = 0;

  // inputDims [n,h,w,i]
  // filterDims [h,w,i,o]
  const inputDesc = {type: nn.TENSOR_FLOAT32, dimensions: inputDims};
  const filterDesc = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3], filterDims[0], filterDims[1], filterDims[2]]};
  const biasDesc = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3]]};
  const intDesc = {type: nn.INT32};

  const input = operandIndex++;
  model.addOperand(inputDesc);
  const filter = operandIndex++;
  model.addOperand(filterDesc);
  const bias = operandIndex++;
  model.addOperand(biasDesc);
  const bias2 = operandIndex++;
  model.addOperand(biasDesc);
  const pad = operandIndex++;
  model.addOperand(intDesc);
  const act = operandIndex++;
  model.addOperand(intDesc);
  const stride = operandIndex++;
  model.addOperand(intDesc);
  const immediateOutput = operandIndex++;
  model.addOperand(inputDesc);
  const output = operandIndex++;
  model.addOperand(inputDesc);

  const filterData = new Float32Array(product(filterDims));
  filterData.fill(filterValue);
  const biasData = new Float32Array(filterDims[3]);
  biasData.fill(biasValue);
  const bias2Data = new Float32Array(filterDims[3]);
  bias2Data.fill(bias2Value);
  model.setOperandValue(filter, filterData);
  model.setOperandValue(bias, biasData);
  model.setOperandValue(bias2, bias2Data);
  model.setOperandValue(pad, new Int32Array([(filterDims[1]-1)/2]));
  model.setOperandValue(act, new Int32Array([nn.FUSE_RELU]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [input, filter, bias, pad, pad, pad, pad, stride, stride, act], [immediateOutput]);
  model.addOperation(nn.CONV_2D, [immediateOutput, filter, bias2, pad, pad, pad, pad, stride, stride, act], [output]);

  model.identifyInputsAndOutputs([input], [output]);
  await model.finish();
  let compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  compilation.setGPUDevice(tf.backend().device);
  await compilation.finish();
  let execution = await compilation.createExecution();

  const inputData = new Float32Array(product(inputDims));
  inputData.fill(inputValue);
  const inputTensor = tf.tensor(inputData, inputDims);
  const outputTensor = tf.zeros(inputDims);

  const inputBuffer = tf.backend().getBuffer(inputTensor.dataId);
  const outputBuffer = tf.backend().getBuffer(outputTensor.dataId);

  // warm up
  executeWebNNForGPU(execution, inputBuffer, outputBuffer);
  // bypass convertAndCacheOnCPU
  let info = tf.backend().tensorMap.get(outputTensor.dataId);
  let data = await tf.backend().getBufferData(info);
  let result = new Float32Array(data);

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    executeWebNNForGPU(execution, inputBuffer, outputBuffer);
    info = tf.backend().tensorMap.get(outputTensor.dataId);
    data = await tf.backend().getBufferData(info);
    result = new Float32Array(data);
  }
  console.log(result);

  const elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv2d/bias/relu x2 in 1 model via WebGPUBuffer elapsed time: ${elapsedTime} ms <br/>`;
}

async function WebNNConvGPUx2Model(inputDims, filterDims) {
  const nn = navigator.ml.getNeuralNetworkContext();
  const model1 = await createWebNNConv(filterValue, biasValue, true);
  const compilation1 = await model1.createCompilation();
  compilation1.setGPUDevice(tf.backend().device);
  compilation1.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation1.finish();
  const execution1 = await compilation1.createExecution();

  const model2 = await createWebNNConv(filterValue, bias2Value, true);
  const compilation2 = await model2.createCompilation();
  compilation2.setGPUDevice(tf.backend().device);
  compilation2.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation2.finish();
  const execution2 = await compilation2.createExecution();

  const inputData = new Float32Array(product(inputDims));
  inputData.fill(inputValue);
  const inputTensor = tf.tensor(inputData, inputDims);
  const immediateTensor = tf.zeros(inputDims);
  const outputTensor = tf.zeros(inputDims);

  const inputBuffer = tf.backend().getBuffer(inputTensor.dataId);
  const immediateBuffer = tf.backend().getBuffer(immediateTensor.dataId);
  const outputBuffer = tf.backend().getBuffer(outputTensor.dataId);

  // warm up
  executeWebNNForGPU(execution1, inputBuffer, immediateBuffer);
  executeWebNNForGPU(execution2, immediateBuffer, outputBuffer);
  // bypass convertAndCacheOnCPU
  let info = tf.backend().tensorMap.get(outputTensor.dataId);
  let data = await tf.backend().getBufferData(info);
  let result = new Float32Array(data);

  const start = performance.now();
  for (let i = 0; i<iterations; i++) {
    executeWebNNForGPU(execution1, inputBuffer, immediateBuffer);
    executeWebNNForGPU(execution2, immediateBuffer, outputBuffer);
    info = tf.backend().tensorMap.get(outputTensor.dataId);
    data = await tf.backend().getBufferData(info);
    result = new Float32Array(data);
  }
  console.log(result);

  const elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv2d/bias/relu x2 in 2 models via WebGPUBuffer elapsed time: ${elapsedTime} ms <br/>`;
}

async function WebNNConvGPUx2WithTf(inputDims, filterDims) {
  const nn = navigator.ml.getNeuralNetworkContext();
  const model = await createWebNNConv(filterValue, biasValue, true);
  const compilation = await model.createCompilation();
  compilation.setGPUDevice(tf.backend().device);
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  const execution = await compilation.createExecution();

  const inputData = new Float32Array(product(inputDims));
  inputData.fill(inputValue);
  const inputTensor = tf.tensor(inputData, inputDims);
  const outputTensor = tf.zeros(inputDims);

  const inputBuffer = tf.backend().getBuffer(inputTensor.dataId);
  const outputBuffer = tf.backend().getBuffer(outputTensor.dataId);

  const filterData = new Float32Array(product(filterDims));
  filterData.fill(filterValue);
  const filter = tf.tensor(filterData, filterDims);
  const biasData = new Float32Array(filterDims[3]);
  biasData.fill(bias2Value);
  const bias = tf.tensor(biasData, [filterDims[3]]);
  executeWebNNForGPU(execution, inputBuffer, outputBuffer);
  let convOutput = tf.conv2d(outputTensor, filter, 1, 'same');
  let addOutput = tf.add(convOutput, bias);
  let reluOutput = tf.relu(addOutput);
  let result = await reluOutput.data();
  let start = performance.now();
  for (let i = 0; i < iterations; i++) {
    executeWebNNForGPU(execution, inputBuffer, outputBuffer);
    convOutput = tf.conv2d(outputTensor, filter, 1, 'same');
    addOutput = tf.add(convOutput, bias);
    reluOutput = tf.relu(addOutput);
    result = await reluOutput.data();
  }
  console.log(result);

  const elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv2d/bias/relu interops with WebGPU conv2d/bias/relu via WebGPUBuffer elapsed time: ${elapsedTime} ms <br/>`;
}

async function test() {
  document.getElementById('backend').innerText = `TF.js sets backend as WebGPU`;
  document.getElementById('size').innerText = `conv input dims: [${inputDims}] and filter dims: [${filterDims}]`;
  await tfConv2d(inputDims, filterDims);
  await WebNNConvCPUWithTf(inputDims, filterDims);
  await WebNNConvGPUWithTf(inputDims, filterDims);
  await WebNNConvGPU(inputDims, filterDims);
  await WebNNConvCPU(inputDims, filterDims);
  await tfConv2dx2(inputDims, filterDims);
  await WebNNConvGPUx2WithTf(inputDims, filterDims);
  await WebNNConvGPUx2Model(inputDims, filterDims);
  await WebNNConvGPUx2(inputDims, filterDims);
  document.getElementById('output').innerHTML += `Done`;
}

async function main() {
  await tf.ready();
  await tf.setBackend('webgpu');
  document.getElementById('start').disabled = false;
  document.getElementById('start').addEventListener('click', () => {test();})
}
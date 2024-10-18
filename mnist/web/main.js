async function loadModel() {
    const model = await tf.loadLayersModel("http://0.0.0.0:8000/model.json",strict=false);
    // const inputTensor = tf.tensor3d([1, 1, 1], [1,28, 28]);



    // Create a grayscale 28x28 image
const grayscaleImage = new Array(28 * 28).fill(0).map(() => Math.random()); // Random grayscale values between 0 and 1

// Convert grayscale to RGB by duplicating the grayscale values across 3 channels
const rgbImage = [];
for (let i = 0; i < grayscaleImage.length; i++) {
  rgbImage.push(grayscaleImage[i])
}

// Create a tensor3d with shape [1, 28, 28, 3] to represent a single RGB image
const inputTensor = tf.tensor4d(rgbImage, [1,28, 28, 1]);


    let predict = model.predict(inputTensor);
    predict.print();  // This will output the prediction result to the console

    console.log(predict.shape,model)
}
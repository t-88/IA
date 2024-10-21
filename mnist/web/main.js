const HOST_DATA = tf.io.fromMemory(
  {
    modelTopology: MODEL_INFO.modelTopology,
    weightSpecs: MODEL_INFO.weightsManifest[0].weights,
    weightData: Uint8Array.from(atob(MODEL_WEIGHTS), c => c.charCodeAt(0)).buffer,
  }
);

var MODEL = undefined;
window.addEventListener("load",async ()  =>  {
  MODEL = await tf.loadLayersModel(HOST_DATA);
})


async function model_predict(img) {
  if(MODEL == undefined) {
    MODEL = await tf.loadLayersModel(HOST_DATA);
  }

  img = tf.tensor(img);
  img = img.reshape([1, 28, 28, 1])

  let predict = await MODEL.predict(img);
  let output = await predict.data();
  let maxIdx = 0;
  let max = output[0];
  for (let i = 1; i < output.length; i++) {
    if (output[i] > max) {
      maxIdx = i;
      max = output[i];
    }
  }
  document.getElementById("res").textContent = "IA Says: " + maxIdx;
}


const GRID_SIZE = 4;
const SCALE = 1;
function to_screen(n) {
  return n / SCALE;
}

var canvas;
var ctx;
var mouse_down = false;
let x, y;
let px, py;


function onMouseDown(e) {
  mouse_down = true;
  px = to_screen(e.offsetX);
  py = to_screen(e.offsetY);
}
function onMouseUp(e) {
  mouse_down = false;

}
function onMouseMove(e) {
  if (mouse_down) {
    x = e.offsetX;
    y = e.offsetY;
    if (e.type == "touchmove") {
      let rect = e.touches[0].target.getBoundingClientRect();
      x = (e.touches[0].clientX - rect.left) / 8;
      y = (e.touches[0].clientY - rect.top) / 8;
    }

    ctx.strokeStyle = "white";
    ctx.fillStyle = "white";
    ctx.fillRect(to_screen(x), to_screen(y), GRID_SIZE, GRID_SIZE)
  }
}


function main() {
  canvas = document.getElementById("canvas");
  ctx = canvas.getContext("2d");
  ctx.scale(SCALE, SCALE)




  document.addEventListener("touchstart", onMouseDown);
  document.addEventListener("touchend", onMouseUp);
  canvas.addEventListener("touchmove", onMouseMove);
  document.addEventListener("mousedown", onMouseDown);
  document.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("mousemove", onMouseMove);


  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 600, 600)
}

main()

const IMG_SIZE = 56;
const DOWN_IMG_SIZE = 28;

function scaleImageData(originalImageData, targetWidth, targetHeight) {
  const targetImageData = new ImageData(targetWidth, targetHeight);
  const h1 = originalImageData.height;
  const w1 = originalImageData.width;
  const h2 = targetImageData.height;
  const w2 = targetImageData.width;
  const kh = h1 / h2;
  const kw = w1 / w2;
  const cur_img1pixel_sum = new Int32Array(4);
  for (let i2 = 0; i2 < h2; i2 += 1) {
    for (let j2 = 0; j2 < w2; j2 += 1) {
      for (let i in cur_img1pixel_sum) cur_img1pixel_sum[i] = 0;
      let cur_img1pixel_n = 0;
      for (let i1 = Math.ceil(i2 * kh); i1 < (i2 + 1) * kh; i1 += 1) {
        for (let j1 = Math.ceil(j2 * kw); j1 < (j2 + 1) * kw; j1 += 1) {
          const cur_p1 = (i1 * w1 + j1) * 4;
          for (let k = 0; k < 4; k += 1) {
            cur_img1pixel_sum[k] += originalImageData.data[cur_p1 + k];
          };
          cur_img1pixel_n += 1;
        };
      };
      const cur_p2 = (i2 * w2 + j2) * 4;
      for (let k = 0; k < 4; k += 1) {
        targetImageData.data[cur_p2 + k] = cur_img1pixel_sum[k] / cur_img1pixel_n;
      };
    };
  };
  return targetImageData;
};

function get_img() {
  let imagedata = ctx.getImageData(0, 0, 56, 56);

  let tmpCanvas = document.createElement('canvas');
  tmpCanvas.getContext("2d").putImageData(imagedata, 0, 0);

  let resizeCanvas = document.createElement('canvas');
  resizeCanvas.width = IMG_SIZE;
  resizeCanvas.height = IMG_SIZE;


  // let resizeCanvas = document.getElementById('tmp-canvas');
  let resizeCtx = resizeCanvas.getContext('2d');
  // imagedata = scaleImageData(imagedata,IMG_SIZE,IMG_SIZE)
  let top = 100;
  let bottom = -1;
  let left = 100;
  let right = -1;
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      if (imagedata.data[4 * (y * IMG_SIZE + x)] != 0) {
        if (top > y) top = y;
        if (left > x) left = x;
        if (right < x) right = x;
        if (bottom < y) bottom = y;
      }
    }
  }
  let width = right - left;
  let height = bottom - top;



  resizeCtx.fillStyle = "black"
  resizeCtx.fillRect(0, 0, IMG_SIZE, IMG_SIZE)
  resizeCtx.putImageData(imagedata, (IMG_SIZE - width) / 2 - left, (IMG_SIZE - height) / 2 - top, left, top, width, height);

  let a = document.createElement('canvas');
  a.width = 28;
  a.height = 28;
  a.getContext("2d").putImageData(scaleImageData(resizeCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE,), 28, 28), 0, 0);


  console.log(a.toDataURL());

  return scaleImageData(resizeCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE,), 28, 28).data.filter((val, idx) => idx % 4 == 0);
}

function predict() {
  let pixels = Array.from(get_img());

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      pixels[y * 28 + x] /= 255.0;
    }
  }

  model_predict(pixels)



}



function clear_canvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 500, 500);
}
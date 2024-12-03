## FasterLivePortrait: Bring portraits to life in Real Time!

**Original repository: [LivePortrait](https://github.com/KwaiVGI/LivePortrait), thanks to the authors for sharing**

### Environment Setup

* First, `apt-get update && apt-get install ffmpeg -y`
* Run `pip install -r requirements.txt`

### Onnxruntime Inference

* First, download the converted onnx model files:

  ```shell
  huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints
  ```
* Follow these steps to install `onnxruntime-gpu` from source:

  ```shell
  git clone https://github.com/microsoft/onnxruntime
  git checkout liqun/ImageDecoder-cuda
  ```

  * Run the following commands to compile, changing `cuda_version` and `CMAKE_CUDA_ARCHITECTURES` according to your machine ( tested on cuda-11.8 + cudnn-8.9.7.29 ) :

  ```shell
  ./build.sh --parallel --build_shared_lib --use_cuda --cuda_version 11.8 --cuda_home /usr/local/cuda-11.8 --cudnn_home /usr/local/cuda-11.8 --config Release --build_wheel --skip_tests --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="80;86" --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc --disable_contrib_ops --allow_running_as_root
  ```

  * Then

  ```shell
  pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl`
  ```
* Test the pipeline using onnxruntime:

  ```shell
  python run.py --src_image assets/examples/source/s10.jpg --dri_video assets/examples/driving/d14.mp4 --cfg configs/onnx_infer.yaml --paste_back
  ```
* For animal:

  ```shell
  cd src/models/XPose/models/UniPose/ops
  python setup.py build install
  cd -
  ```

  ```shell
  python run.py --src_image assets/examples/source/s39.jpg --dri_video assets/examples/driving/d14.mp4 --cfg configs/onnx_infer.yaml --paste_back --animal
  ```

### TensorRT Inference

* [Install TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html):
  ```shell
  wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
  tar -zxvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
  pip install --upgrade pip wheel
  cd TensorRT-8.6.1.6/python
  pip install tensorrt-8.6.1-cp310-none-linux_x86_64.whl tensorrt_dispatch-8.6.1-cp310-none-linux_x86_64.whl tensorrt_lean-8.6.1-cp310-none-linux_x86_64.whl
  cp -r /mnt/afs2/shisheng7/cuda/TensorRT-8.6.1.6/lib/* /usr/lib/x86_64-linux-gnu/
  ```
* Install the grid_sample TensorRT plugin, as the model uses grid sample that requires 5D input, which is not supported by the native grid_sample operator.
  ```shell
  git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin
  # Modify line 30 in CMakeLists.txt to: set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "80;86")
  export PATH=/usr/local/cuda/bin:$PATH
  export CPATH=/usr/local/cuda/include:$CPATH
  export TENSORRT_HOME=/mnt/afs2/shisheng7/cuda/TensorRT-8.6.1.6:$TENSORRT_HOME
  mkdir build && cd build
  cmake .. -DTensorRT_ROOT=$TENSORRT_HOME
  make 
  # remember the address of the .so file, replace `/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so` in `scripts/onnx2trt.py` and `src/models/predictor.py` with your own .so file path
  ```
* Convert all ONNX models to TensorRT, run `sh scripts/all_onnx2trt.sh` and `sh scripts/all_onnx2trt_animal.sh`
* Test the pipeline using tensorrt:
  ```shell
   python run.py \
   --src_image assets/examples/source/s10.jpg \
   --dri_video assets/examples/driving/d14.mp4 \
   --cfg configs/trt_infer.yaml
  ```

### Gradio App

* onnxruntime: `python app.py --mode onnx`
* tensorrt: `python app.py --mode trt`
* The default port is 9870. Open the webpage: `http://localhost:9870/`

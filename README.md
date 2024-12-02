## FasterLivePortrait: Bring portraits to life in Real Time!

`<a href="README.md">`English`</a>` | `<a href="README_CN.md">`中文`</a>`

**Original repository: [LivePortrait](https://github.com/KwaiVGI/LivePortrait), thanks to the authors for sharing**

**Changelog**

- [X] **2024/08/11:** Optimized paste_back speed and fixed some bugs.
  - Used torchgeometry + cuda to optimize the paste_back function, significantly improving speed. Example: `python run.py --src_image assets/examples/source/s39.jpg --dri_video assets/examples/driving/d0.mp4 --cfg configs/trt_infer.yaml --paste_back --animal`
  - Fixed issues with Xpose ops causing errors on some GPUs and other bugs. Please use the latest docker image: `docker pull shaoguo/faster_liveportrait:v3`
- [X] **2024/08/07:** Added support for animal models and MediaPipe models, so you no longer need to worry about copyright issues.
  - Added support for animal models.

    - Download the animal ONNX file: `huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`, then convert it to TRT format.
    - Update the Docker image: `docker pull shaoguo/faster_liveportrait:v3`. Using animal model:`python run.py --src_image assets/examples/source/s39.jpg --dri_video 0 --cfg configs/trt_infer.yaml --realtime --animal`
    - Windows users can download the latest [Windows all-in-one package](https://github.com/warmshao/FasterLivePortrait/releases) from the release page, then unzip and use it.
  - Using MediaPipe model to replace InsightFace

    - For web usage: `python app.py --mode trt --mp` or `python app.py --mode onnx --mp`
    - For local webcam: `python run.py --src_image assets/examples/source/s12.jpg --dri_video 0 --cfg configs/trt_mp_infer.yaml`
- [X] **2024/07/17:** Added support for Docker environment, providing a runnable image.

### Environment Setup

Create a new Python virtual environment and install the necessary Python packages manually.

* First, install [ffmpeg](https://www.ffmpeg.org/download.html)
* Run `pip install -r requirements.txt`
* Then follow the tutorials below to install onnxruntime-gpu or TensorRT. Note that this has only been tested on Linux systems.

### Onnxruntime Inference

* First, download the converted onnx model files:`huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`.
* (Ignored in Docker)If you want to use onnxruntime cpu inference, simply `pip install onnxruntime`. However, cpu inference is extremely slow and not recommended. The latest onnxruntime-gpu still doesn't support grid_sample cuda, but I found a branch that supports it. Follow these steps to install `onnxruntime-gpu` from source:

  * `git clone https://github.com/microsoft/onnxruntime`
  * `git checkout liqun/ImageDecoder-cuda`. Thanks to liqun for the grid_sample with cuda implementation!
  * Run the following commands to compile, changing `cuda_version` and `CMAKE_CUDA_ARCHITECTURES` according to your machine ( tested on cuda-11.8 + cudnn-8.9.7.29 ) :

  ```shell
  ./build.sh --parallel --build_shared_lib --use_cuda --cuda_version 11.8 --cuda_home /usr/local/cuda-11.8 --cudnn_home /usr/local/cuda-11.8 --config Release --build_wheel --skip_tests --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="80;86" --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc --disable_contrib_ops --allow_running_as_root
  ```

  ```shell
  pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl`
  ```
* Test the pipeline using onnxruntime:

  ```shell
  python run.py --src_image assets/examples/source/s10.jpg --dri_video assets/examples/driving/d14.mp4 --cfg configs/onnx_infer.yaml
  ```

* For animal:
  ```shell
  cd src/models/XPose/models/UniPose/ops
  python setup.py build install
  cd -
  ``` 
  ```shell
  python run.py --src_image assets/examples/source/s39.jpg --dri_video assets/examples/driving/d14.mp4 --cfg configs/on
nx_infer.yaml --paste_back --animal
  ```

### TensorRT Inference

* (Ignored in Docker) Install TensorRT. Remember the installation path of [TensorRT](https://developer.nvidia.com/tensorrt).
* (Ignored in Docker) Install the grid_sample TensorRT plugin, as the model uses grid sample that requires 5D input, which is not supported by the native grid_sample operator.
  * `git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin`
  * Modify line 30 in `CMakeLists.txt` to: `set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")`
  * `export PATH=/usr/local/cuda/bin:$PATH`
  * `mkdir build && cd build`
  * `cmake .. -DTensorRT_ROOT=$TENSORRT_HOME`, replace $TENSORRT_HOME with your own TensorRT root directory.
  * `make`, remember the address of the .so file, replace `/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so` in `scripts/onnx2trt.py` and `src/models/predictor.py` with your own .so file path
* Download ONNX model files:`huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`. Convert all ONNX models to TensorRT, run `sh scripts/all_onnx2trt.sh` and `sh scripts/all_onnx2trt_animal.sh`
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

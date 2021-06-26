<!--
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# About
This is an introduction and basic guideline to run inference using NVIDIA Triton inference server

## NVIDIA Triton Inference Server
NVIDIA Triton Inference Server is an open-source inference serving software that simplifies inference serving for an organization by addressing the above complexities. Triton provides a single standardized inference platform which can support running inference on multi-framework models, on both CPU and GPU, and in different deployment environments such as datacenter, cloud, embedded devices, and virtualized environments.

It natively supports multiple framework backends like TensorFlow, PyTorch, ONNX Runtime, Python, and even custom backends. It supports different types of inference queries through advanced batching and scheduling algorithms, supports live model updates, and runs models on both CPUs and GPUs. Triton is also designed to increase inference performance by maximizing hardware utilization through concurrent model execution and dynamic batching. Concurrent execution allows you to run multiple copies of a model, and multiple different models, in parallel on the same GPU. Through dynamic batching, Triton can dynamically group together inference requests on the server-side to maximize performance.

## Open Neural Network Exchange (ONNX)
Open Neural Network Exchange (ONNX) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. Currently we focus on the capabilities needed for inferencing (scoring).

ONNX is widely supported and can be found in many frameworks, tools, and hardware. Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI community. We invite the community to join us and further evolve ONNX.

## TensorRT
TensorRT is an SDK for optimizing trained deep learning models to enable high-performance inference. TensorRT contains a deep learning inference optimizer for trained deep learning models, and a runtime for execution.
After you have trained your deep learning model in a framework of your choice, TensorRT enables you to run it with higher throughput and lower latency.

## PyCUDA
PyCUDA lets you access Nvidia's CUDA parallel computation API from Python.
For more information, check at: https://documen.tician.de/pycuda/

## I. Setup environment and tools
1. ONNX: install ONNX pip and conda packages
```
$ pip install onnx_graphsurgeon 
$ conda install -c conda-forge onnx
```
Note: If convert pth to onnx get error (libstdc++.so.6: version `GLIBCXX_3.4.22' not found), fix by run below commands:
```
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test 
$ sudo apt-get update 
$ sudo apt-get install gcc-4.9 
$ sudo apt-get install --only-upgrade libstdc++6 
```
2. TensorRT (for detail install instruction, check at: https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#install)
- Login and download from https://developer.nvidia.com/tensorrt
- Install downloaded deb package as guide (https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian)
- Install pip packages (https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip)
```
$ python3 -m pip install --upgrade setuptools pip 
```
- You should now be able to install the nvidia-pyindex module.
```
$ python3 -m pip install nvidia-pyindex 
$ python3 -m pip install --upgrade nvidia-tensorrt 
```
3. PyCUDA (for details, check at: https://wiki.tiker.net/PyCuda/Installation/Linux/#step-1-download-and-unpack-pycuda)
-  Step 1: Download source of pip package tar.gz and unpack PyCUDA from https://pypi.org/project/pycuda/#files
```
$ tar xfz pycuda-VERSION.tar.gz
```
- Step 2: Install Numpy
PyCUDA is designed to work in conjunction with numpy, Python's array package. Here's an easy way to install it, if you do not have it already:
```
$ cd pycuda-VERSION
$ su -c "python distribute_setup.py" # this will install distribute
$ su -c "easy_install numpy" # this will install numpy using distribute
```
- Step 3: Build PyCUDA
Next, just type:
```
Install make if needed:
$ sudo apt-get install -y make

Start building:
$ cd pycuda-VERSION # if you're not there already
$ python configure.py --cuda-root=/where/ever/you/installed/cuda
$ su -c "make install"
```
- Step 4: Test PyCUDA
If you'd like to be extra-careful, you can run PyCUDA's unit tests:
```
$ cd pycuda-VERSION/test
$ python test_driver.py
```
## II. Prepare Triton server docker image, akaOCR source repo or client docker image
Pull repo, image, and prepare models (Where <xx.yy> is the version of Triton that you want to use):
```
$ sudo docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
<!-- $ git clone https://github.com/triton-inference-server/server.git -->
$ git clone https://gitlab.com/cuongvt/ocr-components.git
Run the .sh script to convert model into target formats, prepare Model Repo and start Triton server container:
$ cd ocr-components/akaocr/tools/inference
$ sh prepare.sh
Convert source model into target formats and copy into Triton's Model Repository successfully.
```
## III. Run the server and client to infer (included in .sh script):
Run server in container and client in cmd
```
$ sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v <full_path_to/data/model_repository>:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models

Kimnh3 developing on server:
$ sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/maverick911/repo/ocr-components-triton/akaocr/data/model_repository:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models

+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| detec_pt             | 1       | READY  |
| detec_trt            | 1       | READY  |
....
I0611 04:10:23.026207 1 grpc_server.cc:4062] Started GRPCInferenceService at 0.0.0.0:8001
I0611 04:10:23.036976 1 http_server.cc:2987] Started HTTPService at 0.0.0.0:8000
I0611 04:10:23.080860 1 http_server.c9:2906] Started Metrics Service at 0.0.0.0:8002
```
2. Infer by client in cmd (this repo), for ex:
```
$ cd ocr-components/akaocr/tools/inference
$ python infer_triton.py -m='detec_pt' -x=1 --input='../../test/images/image_1.jpg'
Reading input image from file ../../test/images/image_1.jpg
Running inference using Triton for akaOCR detec engine
Request 1, batch size 1

Inference using Triton server: 
Total execution time = 1.774 sec
Max memory used by tensors = 0 bytes
PASS
../../data/infer_result/image_1_detec_triton.jpg
```
```
$ python infer_triton.py -m='detec_trt' -x=1 --input='../../test/images/image_1.jpg' -i='grpc' -u='localhost:8001'
Inference using Triton server: 
Total execution time = 0.637 sec
Max memory used by tensors = 201001984 bytes
PASS
Infer successfully, saved result at:  ../../data/infer_result/image_1_detec_triton.jpg
```
-------
Run server in container and client sdk in container:
1. Start the server side:
```
$ sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/maverick911/repo/ocr-components-triton/akaocr/data/model_repository:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models

+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| akaocr_detec         | 1       | READY  |
| akaocr_detec_trt     | 1       | READY  |
....
I0611 04:10:23.026207 1 grpc_server.cc:4062] Started GRPCInferenceService at 0.0.0.0:8001
I0611 04:10:23.036976 1 http_server.cc:2987] Started HTTPService at 0.0.0.0:8000
I0611 04:10:23.080860 1 http_server.c9:2906] Started Metrics Service at 0.0.0.0:8002
```
2. Start client image to start inferencing (shell), mount client src into container:
```
$ sudo docker run -it --rm --net=host -v <full_path/to/ocr-components>:/workspace/client nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```
3. Use infer_triton.py as example above to run.

## Note:
- Trained model which saved by torch.save (usually .pth) must be convert into torchscript by torch.jit.save (into model.pt as default name of Triton).
- TensorRT need to be installed with the same version as used in Triton server Docker image, so that the engine created by TensorRT can be loaded into Model Repo of Triton. For ex, in Triton server version 21.05, TensorRT's version is 7.2.3.4
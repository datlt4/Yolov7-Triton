# Yolov7-Triton

## Prepare

- In this my example, I used models from 2 repository: [yolov4-triton-tensorrt - isarsoft](https://github.com/isarsoft/yolov4-triton-tensorrt/tree/b42dd1faf50e516b3943eafd3c15408f536bf1c5) and [Yolov4-AlphaPose-MOT-Trt - LuongTanDat](https://github.com/LuongTanDat/Yolov4-AlphaPose-MOT-Trt)
- Environment:

> GPU: Nvidia GeForce RTX3060<br>
> Driver version: 510.73.05<br>
> CUDA version: 11.3<br>
> CuDNN version:8.2.1<br>
> TensorRT version: 8.0.3.4

### Build TensorRT engines

- Follow below instructions: [Yolov4-Tensorrtx](https://github.com/isarsoft/yolov4-triton-tensorrt/tree/b42dd1faf50e516b3943eafd3c15408f536bf1c5#build-tensorrt-engine), [Yolov4-darknet, Deepsort, Alphapose](https://github.com/LuongTanDat/Yolov4-AlphaPose-MOT-Trt#build-project)

- Or pull converted models use dvc if you are using the same hardware specification: `dvc pull`, send to email `20150837@student.hust.edu.vn` to request access

## Start triton server

```bash
docker run --gpus all --rm \
    --name=triton-server \
    --shm-size=1g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p8000:8000 -p8001:8001 -p8002:8002 \
    -v$(pwd)/models:/models \
    nvcr.io/nvidia/tritonserver:21.10-py3 tritonserver \
    --model-repository=/models \
    --strict-model-config=false \
    --grpc-infer-allocation-pool-size=16 \
    --log-verbose 1
```

```
+------------+---------+--------+
| Model      | Version | Status |
+------------+---------+--------+
| yolov7-trt | 1       | READY  |
+------------+---------+--------+
.
.
.
I0720 20:06:26.600645 1 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0720 20:06:26.600796 1 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0720 20:06:26.641779 1 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

### `Curl` info models

```bash
$ curl http://127.0.0.1:8000/v2/models/yolov7-trt
{"name":"yolov7-trt","versions":["1"],"platform":"tensorrt_plan","inputs":[{"name":"images","datatype":"FP32","shape":[-1,3,608,608]}],"outputs":[{"name":"output","datatype":"FP32","shape":[-1,22743,12]}]}
```

## python client

### Install dependencies

```bash
cd ${WORKING_DIR}/client/python/

conda env create -f environment.yml
conda activate triton
$(which python) -m pip install nvidia-pyindex tritonclient[all] requests Flask flask-cors pyopenssl
```

### Run examples

```bash
python request_yolov7_trt.py
```

### Run API

```bash
python app.py
```

<!-- 
## [C++ client](https://github.com/olibartfast/object-detection-inference)

### Install dependencies

```bash
sudo apt install libssl-dev
python3 -m pip install --user grpcio-tools
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson
cmake .
make
sudo make install
cd -

git clone https://github.com/triton-inference-server/client.git -b r21.05
cd client
sed -i 's/byte_contents/bytes_contents/' src/c++/library/grpc_client.cc
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_PERF_ANALYZER=ON -DTRITON_ENABLE_PYTHON_HTTP=ON -DTRITON_ENABLE_PYTHON_GRPC=ON -DTRITON_ENABLE_GPU=ON -DTRITON_ENABLE_EXAMPLES=ON -DTRITON_ENABLE_TESTS=ON ..
make cc-clients python-clients -j$(nproc)
TritonClientThirdParty_DIR=$(pwd)/third-party
TritonClientBuild_DIR=$(pwd)/install
```

### Build and run examples

```bash
cd ${WORKING_DIR}/client/c++/
mkdir build
cd build
cmake ..
make
./yolov4-triton-cpp-client
``` -->

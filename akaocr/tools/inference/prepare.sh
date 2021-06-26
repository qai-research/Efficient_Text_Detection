# Before begin,
# please rename 2 initial .pth model in experiment folder into format: <name>_detec.pth and <name>_recog.pth
export exp='test' # Set experiment name, default='test'
export failed=0 # To check if cmd run completed
export CUDA_VISIBLE_DEVICES=0 # Set specific GPU for TensorRT engine optimized for, start from 0
# Create model repository for Triton
mkdir -p ../../data/model_repository/detec_pt/1 ../../data/model_repository/detec_onnx/1 ../../data/model_repository/detec_trt/1 \
    ../../data/model_repository/recog_pt/1 ../../data/model_repository/recog_onnx/1 ../../data/model_repository/recog_trt/1


# I. Convert pth model into (Torch JIT, ONNX, TensorRT)
python ../converters/pth2jit.py --exp=$exp || export failed=1
python ../converters/pth2onnx.py --input ../../data --exp test --output ../../data || export failed=1
# python ../converters/onnx2trt.py || export failed=1
/usr/src/tensorrt/bin/trtexec --onnx=../../data/exp_detec/test/detec_onnx.onnx --explicitBatch --workspace=5000 --minShapes=input:1x3x256x256 --optShapes=input:1x3x700x700 --maxShapes=input:1x3x1200x1200 --buildOnly --saveEngine=../../data/exp_detec/test/detec_trt.engine


# II. Copy models into Model Repo for Triton server
# pt
cp ../../data/exp_detec/test/detec_pt.pt ../../data/model_repository/detec_pt/1/detec_pt.pt
cp ../../data/exp_recog/test/recog_pt.pt ../../data/model_repository/recog_pt/1/recog_pt.pt

# onnx
cp ../../data/exp_detec/test/detec_onnx.onnx ../../data/model_repository/detec_onnx/1/detec_onnx.onnx

# trt
cp ../../data/exp_detec/test/detec_trt.engine ../../data/model_repository/detec_trt/1/detec_trt.plan

if [ ${failed} -ne 0 ]; then
        echo "Prepare failed, check error on the terminal history above..."
      else
        echo "Convert source model into target formats and copy into Triton's Model Repository successfully."
      fi

# III. Start Triton server image in container, mount Model Repo prepared into container volume
# Update the full path to data/model_repository follow deploy server path: "-v <full_path_to>/ocr-components-triton/akaocr/data/model_repository:/models"
sudo docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/maverick911/repo/ocr-components-triton/akaocr/data/model_repository:/models nvcr.io/nvidia/tritonserver:21.05-py3 tritonserver --model-repository=/models
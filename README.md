# deep_view_syn

Requirement:

Pytorch 1.7

cv2

numpy

matplotlib

torchvision

json


跑训练：
python main_lf_syn.py

DATA_FILE = "/home/yejiannan/Project/LightField/data/lf_syn"
DATA_JSON = "/home/yejiannan/Project/LightField/data/data_lf_syn_full.json"
OUTPUT_DIR = "/home/yejiannan/Project/LightField/outputE/lf_syn_full"

这里要重新配置一下数据的位置

Video generate:
ffmpeg -y -r 30 -i view_%04d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p ../train.mp4
trtexec --onnx=net@256x256x2.onnx --fp16 --saveEngine=net@256x256x2.trt --workspace=4096
import io
import json
import os
import torch

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./ResNet.pt", map_location=device)

print(model.eval())

img_class_map = None
mapping_file_path = 'index_to_name.json'                  # 사람이 읽을 수 있는 ImageNet 클래스 이름
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)

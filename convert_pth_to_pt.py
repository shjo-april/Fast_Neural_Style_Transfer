# Copyright (C) 2022 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from core import networks
from tools.general import io_utils

parser = io_utils.Parser()
parser.add('pth_path', './experiments/models/Transformer@vc_monariza/ep=2.pth', str)
parser.add('pt_path', './weights/Transformer@vc_monariza.pt', str)
args = parser.get_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

image = Image.open('./samples/SSAL_1.jpg').convert('RGB')
image = transform(image)
image = image.unsqueeze(0)

model = networks.Transformer(in_channels=3)
model.load_state_dict(torch.load(args.pth_path, map_location='cpu'))

model.eval()

traced_script_module = torch.jit.trace(model, image)
traced_script_module._save_for_lite_interpreter(args.pt_path)

# traced_model_load = torch.jit.load(path)
# traced_output = traced_model_load(image)
# print(traced_output)

# pymodel_output = model(image)
# print(pymodel_output)

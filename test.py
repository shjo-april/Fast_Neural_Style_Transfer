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
parser.add('image_path', './samples/SSBO_1.jpg', str)
args = parser.get_args()

# preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

image = Image.open(args.image_path).convert('RGB')
image = transform(image)
image = image.unsqueeze(0).cuda()

# define a model
model = networks.Transformer(in_channels=3)

model.eval()
model.cuda()

for tag in [
        'Transformer@edtaonisl',
        'Transformer@vc_monariza',
        'Transformer@vg_starry_night',
        'Transformer@wave',
    ]:
    # load a weight
    model.load_state_dict(torch.load(f'./experiments/models/{tag}/ep=1.pth'))

    # inference
    stylized_image = model(image)

    # postprocessing
    stylized_image = stylized_image.cpu().detach().numpy()
    stylized_image = stylized_image[0].transpose((1, 2, 0))

    stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)
    stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)

    # visualize
    cv2.imshow('Demo', stylized_image)
    cv2.waitKey(0)
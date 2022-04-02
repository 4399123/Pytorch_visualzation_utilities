from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn,fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image
import torch
import random
import numpy as np
# from glob import glob
from pathlib import Path

Colors=[ [ random.randint(0,255) for _ in range(3)] for _ in range(91)]
p=Path.cwd()
pic_paths=list(p.glob('pic/*.jpg'))





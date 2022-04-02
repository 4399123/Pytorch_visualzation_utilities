from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms.functional import convert_image_dtype
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torch
import random
import numpy as np
from show_images import  show
Colors=[ [ random.randint(0,255) for _ in range(3)] for _ in range(91)]

pic1_path='1.jpg'
pic2_path='2.jpg'
pic1=read_image(pic1_path)
pic2=read_image(pic2_path)

batch_int=torch.stack([pic1,pic2])
print(batch_int.shape)
batch=convert_image_dtype(batch_int,dtype=torch.float)

model=fasterrcnn_mobilenet_v3_large_fpn(pretrained=True,)
model=model.eval()
outputs=model(batch)

dog_with_boxes=[draw_bounding_boxes(dog_int,boxes=output['boxes'][output['scores']>0.8],width=4)
    for dog_int,output in zip(batch_int,outputs)]
show(dog_with_boxes)


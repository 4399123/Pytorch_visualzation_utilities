from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn,fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image
from PIL.ImageDraw import ImageDraw
import torch
import random
import numpy as np
# from glob import glob
from pathlib import Path
from show_images import PIL_show

Colors=[ [ random.randint(0,255) for _ in range(3)] for _ in range(91)]
p=Path.cwd()
pic_paths=list(p.glob('pic/*.jpg'))

batch_int_np=list()
for pic_path in pic_paths:
    image=Image.open(pic_path)
    image=torch.tensor(np.array(image).transpose(2,0,1))
    batch_int_np.append(image)

batch_int=torch.stack(batch_int_np)
batch=convert_image_dtype(batch_int,dtype=torch.float)

model=fasterrcnn_resnet50_fpn(pretrained=True)
model=model.eval()
outputs=model(batch)

rec_imgs=[]
for img,output in zip(batch_int,outputs):
    boxes = np.array(output['boxes'][output['scores'] > 0.8].detach())
    labels = np.array(output['labels'][output['scores'] > 0.8].detach())
    img = np.ascontiguousarray(np.array(img.detach()).transpose(1, 2, 0).astype(np.uint8))
    img=Image.fromarray(img).convert('RGB')
    a=ImageDraw(img)
    for box, label in zip(boxes, labels):
        a.rectangle(((int(box[0]), int(box[1])),(int(box[2]), int(box[3]))), fill=None, outline=tuple(Colors[label]), width=5)
    rec_imgs.append(img)

PIL_show(rec_imgs)






from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn,fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype
import cv2
import torch
import random
import numpy as np

Colors=[ [ random.randint(0,255) for _ in range(3)] for _ in range(91)]

pic1_path='1.jpg'
pic2_path='2.jpg'

pic1=torch.from_numpy(np.array(cv2.imread(pic1_path)[:,:,::-1])).permute(2,0,1)
pic2=torch.from_numpy(np.array(cv2.imread(pic2_path)[:,:,::-1])).permute(2,0,1)

batch_int=torch.stack([pic1,pic2])
print(batch_int.shape)
batch=convert_image_dtype(batch_int,dtype=torch.float)

model=fasterrcnn_resnet50_fpn(pretrained=True)
model=model.eval()
outputs=model(batch)

for dog_int,output in zip(batch_int,outputs):
    boxes=np.array(output['boxes'][output['scores']>0.8].detach())
    labels = np.array(output['labels'][output['scores'] > 0.8].detach())
    dog_int=np.ascontiguousarray(np.array(dog_int.detach()).transpose(1,2,0).astype(np.uint8)[:,:,::-1])
    for box,label in zip(boxes,labels):
        cv2.rectangle(dog_int,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),Colors[label],2)
    cv2.imshow('11',dog_int)
    cv2.waitKey(0)

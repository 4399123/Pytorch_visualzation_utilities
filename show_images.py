import torch
import  numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs,list):
        imgs=[imgs]
    _,axs=plt.subplots(ncols=len(imgs),squeeze=False)
    for i,img in enumerate(imgs):
        img=img.detach()
        img=F.to_pil_image(img)
        axs[0,i].imshow(np.array(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def PIL_show(imgs):
    num=len(imgs)
    ncols=2
    nrows=num//2
    if nrows==0:
        nrows=1
    _, axs = plt.subplots(nrows=nrows,ncols=ncols, squeeze=False)
    n=0
    for nrow in range(nrows):
        for ncol in range(ncols):
            axs[nrow, ncol].imshow(imgs[n])
            n+=1
    plt.savefig('result.png',dpi=800)
    plt.show()

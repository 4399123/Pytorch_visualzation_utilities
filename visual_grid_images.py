from show_images import show
from torchvision.utils import make_grid
from torchvision.io import read_image

pic1_path='1.jpg'
pic2_path='2.jpg'
pic1=read_image(pic1_path)
pic2=read_image(pic2_path)

grid=make_grid([pic1,pic2])
show(grid)

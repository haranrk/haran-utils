import torch
import torchvision
import sys
from torchvision import transforms
root = 'more-flare'

dataset = torchvision.datasets.ImageFolder(root,transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset)
data_iter = iter(loader)
shapes = {}
error_count = 0
for i in range(len(dataset)):
    try:
        im,_ = data_iter.next()
    except OSError:
        error_count +=1
    shape = (im.shape[-2], im.shape[-1])
    if shape not in shapes.keys():
        shapes[shape] = 1
    else:
        shapes[shape] +=1
    sys.stdout.write(f"\r {str(shapes)} {error_count}")
    sys.stdout.flush()
    

print(shapes)



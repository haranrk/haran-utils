'''
Find and print all the resolutions present in the dataset
'''
import torch
import torchvision
import sys
from torchvision import transforms
root = sys.argv[1]

dataset = torchvision.datasets.ImageFolder(root,transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset)
data_iter = iter(loader)
shapes = {}
error_list = []
for i in range(len(dataset)):
    try:
        im,_ = data_iter.next()
    except OSError as e:
        error_list.append(e)
    shape = (im.shape[-2], im.shape[-1])
    if shape not in shapes.keys():
        shapes[shape] = 1
    else:
        shapes[shape] +=1
    if i%1000:
        sys.stdout.write(f"\r {str(shapes)} {len(error_list)}")
        sys.stdout.flush()
    
print(error_list)
import pdb; pdb.set_trace()



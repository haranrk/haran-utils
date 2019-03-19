'''
Script to center crop and resize images
Positional Args:
    output_dir: Dir to which the images would be output
    Any number of input images
'''
import os, sys
from PIL import Image

image_size = (512,512) 

out_dir = sys.argv[1]
for in_path in sys.argv[2:]:
    out_path = os.path.join(out_dir, os.path.split(in_path)[-1])
    im = Image.open(in_path)
    print(f"{in_path}:{im.size} -> {out_path}:{image_size}")
    w, h = im.size
    min_dim = min(w,h)
    left = w/2 - min_dim/2
    right = w/2 + min_dim/2
    upper = h/2 - min_dim/2
    lower = h/2 + min_dim/2
    im = im.crop((left,upper,right, lower))
    im.thumbnail(image_size)
    im.save(out_path)

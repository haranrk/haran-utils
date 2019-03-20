'''
Moves images from source to target_dir if they have a greater resolution than specified
Positional arguments:
    source_dir
    target_dir
'''
from PIL import Image
import os, sys, shutil

img_dir = sys.argv[1]
target_dir = sys.argv[2]
total = 0
success = 0
for img_path in os.listdir(img_dir):
    total +=1
    img_path = os.path.join(img_dir, img_path)
    im = Image.open(img_path)
    if min(im.size) > 512:
        success +=1
        print(f"{img_path}:{im.size} -> {target_dir}")
        shutil.move(img_path, target_dir)
    
    
print(f"{success}/{total}")

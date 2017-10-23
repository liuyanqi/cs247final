import os
import json
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ImageDir='/home/jasmine/Downloads/'
dataDir='/home/jasmine/Desktop/cshw/final_proj'
dataType='train2014'
annFile='{}/annotations/captions_{}.json'.format(dataDir,dataType)
# initialize COCO api for instance annotations

data = json.load(open('coco_raw.json', 'r'))
coco=COCO(annFile)

img = data[0]
I = Image.open(ImageDir + img['file_path'])
print(img['captions'])

plt.axis('off')
plt.imshow(I)
plt.show()
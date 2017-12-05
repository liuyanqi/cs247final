"""
This script should be run from root directory of this codebase:
https://github.com/tylin/coco-caption
"""

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys



annFile = 'annotations/captions_flickr8k.json'
coco = COCO(annFile)
resFile = './results/result4.json'
cocoRes = coco.loadRes(resFile)
cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.params['image_id'] = cocoRes.getImgIds()
cocoEval.evaluate()

# create output dictionary
for metric, score in cocoEval.eval.items():
    print '%s: %.3f'%(metric, score)
# serialize to file, to be read from Lua
print('done')


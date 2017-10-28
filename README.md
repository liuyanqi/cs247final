# cs247final

## Database needed:
http://cocodataset.org/#download \
Download train data(http://images.cocodataset.org/zips/train2014.zip) and val data (http://images.cocodataset.org/zips/val2014.zip)\

## preprocessing
The preprocessing code is adapted from the original project repo (https://github.com/karpathy/neuraltalk2/blob/master/coco/coco_preprocess.ipynb).It break the data into a json file in the format of 

0 'file_path' ,'captions'\
1 'file_path' ,'captions'\
...

## rcnn.py
by specify directory of the training image, it is able to display the image and print out the corresponding captions

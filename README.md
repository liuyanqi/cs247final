# cs247final

## Database needed:
### MSCOCO (largest)
http://cocodataset.org/#download \
Download train data(http://images.cocodataset.org/zips/train2014.zip) and val data (http://images.cocodataset.org/zips/val2014.zip) \

###Flicker8k dataset
Download image data: http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip \
Download text data: http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip

## preprocessing.py
The preprocessing code is uses the vgg16 network adapted from(https://github.com/machrisaa/tensorflow-vgg) and you need to download the pretrained vgg16.npy file from the repo.
Preprocessing.py scale the image to 224x224 and output the 4096D feature into the output file
### output data format
	filename: 4096D feature
## extract.py
extract.py reading in captions for each image, pad them to be the max length and save in dict with format: \
	filename: [cap1, cap2, ..., cap5]


import os
import sys
import urllib

url = "http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip"
save_path = "./data/captions_train-val2014.zip"

if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)

with zipfile.ZipFile('.data/captions_train-val2014') as zip_file:
    zip_file.extractall('.data/')


url = "http://images.cocodataset.org/zips/train2014.zip"
save_path = "./data/train2014.zip"

if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)

with zipfile.ZipFile('./data/train2014.zip') as zip_file:
    zip_file.extractall('.data/')


url = "http://images.cocodataset.org/zips/val2014.zip"
save_path = "./data/val2014.zip"

if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)

with zipfile.ZipFile('./data/val2014.zip') as zip_file:
    zip_file.extractall('.data/')

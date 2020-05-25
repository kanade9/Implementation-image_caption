import os
import sys
import urllib
import zipfile

url = 'http://images.cocodataset.org/zips/val2017.zip'
save_path = "./data/val2017.zip"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
urllib.request.urlretrieve(url, save_path)

with zipfile.ZipFile('data/val2017.zip') as zip_file:
    zip_file.extractall('data/')

url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
save_path = "./data/annotations_trainval2017.zip"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
urllib.request.urlretrieve(url, save_path)

with zipfile.ZipFile('data/annotations_trainval2017.zip') as zip_file:
    zip_file.extractall('data/')

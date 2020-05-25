import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

base = os.path.dirname(os.path.abspath(__file__))

val_path = base + "/data/annotations/instances_val2017.json"
coco = COCO(val_path)

ann_path = base + "/data/annotations/captions_val2017.json"
coco_caps = COCO(ann_path)

ids = list(coco.anns.keys())
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']

img_dict = coco.loadImgs(img_id)[0]
img_path = base + "/data/val2017/" + str(img_dict['file_name'])

print(img_dict['coco_url'])
img = plt.imread(img_path)
plt.imshow(img)

annIds = coco_caps.getAnnIds(imgIds=img_dict['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)

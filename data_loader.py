import os
import nltk
import torch
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO
from build_vocab import Vocabulary
import torchvision.transforms as transforms


class CocoDataset(data.Dataset):

    def __init__(self, root, json, vocab, transform=None):
        """
        root: 画像のrootディレクトリ
        json: cocoのannotationファイルのパス
        vocab: vocabulary wrapper
        transform: 画像の変形
        """

        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        # imageとcaptionのペアを返す

        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[idnex]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # captionの文字をワードのidxに変換する
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])

    def collate_fn(data):
        # (image,caption)というタプルになっているリストからミニバッジのテンソルを作成する

        # dataのimageは(3,256,256)のようなtensor、captionは可変の次元

        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # タプル型の3Dテンソルを4Dのテンソルにすることで画像をマージする
        images = torch.stack(images, 0)

        # タプル型の1Dテンソルを2Dのテンソルにすることでcaptionをマージする
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lenghts[i]
            targets[i, :end] = cap[:end]

        # images:(batch_size,3,256,256)
        # targets:(batch_size,padded_length)
        # lengths:->list captionの長さ(paddingされてるかも)
        return images, targets, lengths

    def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):

        coco = CocoDataset(root=root, json=json, vocab=vocab, transform=transform)

        data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size, shuffle=shuffle,
                                                  num_workers=num_workers, collate_fn=collate_fn)

        return data_loader

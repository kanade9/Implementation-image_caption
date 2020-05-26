import torch
import torch.nn as nn
from torchvision import transforms
import nltk

nltk.download('punkt')
from pycocotools.coco import COCO
from data_loader import get_loader
import torch
import hydra


@hydra.main(config_path="config/config.yaml")
def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform_train = transforms.Compose([
        # 画像を256x256にする
        transforms.Resize(cfg.resize_size),
        # 224x224の領域をランダムにクロップ処理する
        transforms.RandomCrop(cfg.crop_size),
        # 0.5の割合で水平に反転させる
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg.img.mean, cfg.img.std)
    ])

    data_loader = get_loader(transform=transform_train, model='train')

import os
import torch
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(iamge_path, transform=None):
    image = Image.open(iamge_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


@hydra.main(config_path="config/config.yaml")
def main(cfg):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.image.mean, cfg.image.std)])

    with open(cfg.train.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # モデルの構築
    encoder = EncoderCNN(cfg.train.embed_size).eval()
    decoder = DecoderRNN(cfg.train.embed_size, cfg.train.hidden_size, len(vocab), cfg.train.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 学習済みモデルのパラメータを読み込む
    encoder.load_state_dict(torch.load(cfg.train.encoder_path))
    decoder.load_state_dict(torch.load(cfg.train.decoder_path))

    # 画像の準備
    image = load_image(cfg.sample.image_path, transform)
    image_tensor = image.to(device)

    # 入力した画像からキャプションを生成する
    fwature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # word_idsをwordに変換する
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    print(sentence)
    image = Image.open(cfg.sample.image_path)
    plt.imshow(np.asarray(image))
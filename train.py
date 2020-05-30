import os
import torch
import hydra
import pickle
import numpy as np
import torch.nn as nn
from torchvision import transforms
from build_vocab import Vocabulary
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    # modelのディレクトリ作成
    if not os.path.exists(hydra.utils.to_absolute_path(cfg.train.model_path)):
        os.makedirs(hydra.utils.to_absolute_path(cfg.train.model_path))

    # 画像の前処理と正規化を行う
    transform = transforms.Compose([
        transforms.RandomCrop(cfg.image.resize_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg.image.mean, cfg.image.std)
    ])

    with open(hydra.utils.to_absolute_path(cfg.train.vocab_path), 'rb') as f:
        vocab = pickle.load(f)

    # data_loaderの読み込み
    data_loader = get_loader(hydra.utils.to_absolute_path(cfg.resize.image_dir),
                             hydra.utils.to_absolute_path(cfg.train.caption_path), vocab, transform,
                             cfg.train.batch_size, shuffle=True,
                             num_workers=cfg.train.num_workers)

    # modelの構築
    encoder = EncoderCNN(cfg.train.embed_size).to(device)
    decoder = DecoderRNN(cfg.train.embed_size, cfg.train.hidden_size, len(vocab), cfg.train.num_layers).to(device)

    # lossとoptimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.train.learning_rate)

    # train
    total_step = len(data_loader)
    print(data_loader)

    for epoch in range(cfg.train.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            print(images,captions,lengths)
            # ミニバッジデータセットのセット
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % cfg.train.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, cfg.train.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

            # modelをcheckpointごとにSaveする
            if (i + 1) % cfg.train.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    hydra.utils.to_absolute_path(cfg.train.model_path), 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))

                torch.save(encoder.state_dict(), os.path.join(
                    hydra.utils.to_absolute_path(cfg.train.model_path), 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))


if __name__ == '__main__':
    main()

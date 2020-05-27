import nltk
# オブジェクトをバイナリ列などに変換してファイルに書き込む
import pickle
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['capiton'])
        # captionをnltkでtokenizaする
        tokens = nltk.tokenize.word_tokenize((caption.lower()))
        counter.update()

        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))

    # vocab frequencyの設定。frequencyがthreshold未満だったら無視する
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # vocab wrapperの作成と特別なtokenの付加処理

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # wordをvocabに入れる
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab



@hydra.main(config_path="config/config.yaml")
def main(cfg):
    vocab = build_vocab(json=cfg.vocab.caption_path, threshold=cfg.vocab.threshold)
    vocab_path = cfg.vocab.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format((len(vocab))))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

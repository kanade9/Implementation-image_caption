import torch
import hydra
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        # pretrainされたresnet512のモデルを読み込み、fc層に送る
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        # 最後のfc層を削除する
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # 入力された画像から特徴ベクトルを抽出する
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size,vocab_size,num_layers, max_seq_length=20):
        # ハイパラをセットして層を組み立てる
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        # 画像のベクトル特徴量をデコードしてキャプションを生成する
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        # 貪欲法を使って与えられた画像からキャプションを生成する(beam searchを使うと改善するってどこかの記事に書いてあった)
        # 別のアルゴリズムでも検討してみたい。

        sampled_ids = []
        inputs = features.unsqueeze(1)
        # inputs.shape=[1,1,256]
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

    def greedy_decode(self,features,states=None):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        Hはhidden sizeを表す
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        batch_size, seq_len = 1,self.max_seg_length
        decoded_batch = torch.zeros((batch_size, self.max_seg_length))
        decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)])
        inputs = features.unsqueeze(1)
        for t in range(self.max_seg_length):
            # decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            decoder_output=outputs

            topv, topi = decoder_output.topk(1)  # get candidates
            topi = topi.view(-1)
            inputs = self.embed(topi)
            inputs = inputs.unsqueeze(1)
            decoded_batch[:, t] = topi

            decoder_input = topi.detach().view(-1, 1)

        return decoded_batch


# モデルの確認

# embed_size = 512
# encoder = EncoderCNN(embed_size)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# encoder = encoder.to(device)
# print(encoder)

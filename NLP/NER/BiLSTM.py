import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from HMM import load_data
from copy import deepcopy
from utils import save_model, load_model
from evaluation import Metrics


def cal_loss_func(logits, targets, tag2id):
    """
    计算损失
    :param logits: [B, L, out_size]
    :param targets: [B, L]
    :param tag2id: [B]
    :return:
    """
    PAD = tag2id.get("<pad>")
    assert PAD is not None

    mask = (targets != PAD)
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)

    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)

    return loss


def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    # 这里的indices构建了一个映射关系
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


def tensorized(batch, maps):
    # 没有很理解这里的张量化想要做什么
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    # 这里乘PAD的意义是什么呢
    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)

    lengths = [len(l) for l in batch]

    return batch_tensor, lengths


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size, batch_first=True,
                              bidirectional=True)
        self.lin = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.lin(rnn_out)

        return scores

    def test(self, sents_tensor, lengths, _):
        logits = self.forward(sents_tensor, lengths)
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids


class BILSTMModel(object):
    def __init__(self, vocab_size, out_size, crf=True):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.emb_size = 128
        self.hidden_size = 128
        self.crf = crf
        self.model = BiLSTM(vocab_size, self.emb_size, self.hidden_size, out_size).to(self.device)
        self.cal_loss_func = cal_loss_func

        self.epoches = 30
        self.batch_size = 64
        self.learning_rate = 0.001
        self.print_step = 5

        # 初始化优化器
        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, word_lists, tag_lists, dev_word_lists, dev_tag_lists,
              word2id, tag2id):
        # 按照长度对内容进行排序
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)

        B = self.batch_size
        for e in range(1, self.epoches+1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind: ind+B]
                batch_tags = tag_lists[ind: ind+B]

                losses += self.train_step(batch_sents, batch_tags, word2id, tag2id)

                if self.step % 5 == 0:
                    total_step = (len(word_lists) // B + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            val_loss = self.validate(
                dev_word_lists, dev_tag_lists, word2id, tag2id
            )
            print("Epoch {}, Val Loss: {:.4f}".format(e, val_loss))

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1

        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, lengths = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)

        # forward
        scores = self.model(tensorized_sents, lengths)

        # 计算损失
        self.optim.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        loss.backward()
        self.optim.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备Batch数据
                batch_sents = dev_word_lists[ind: ind+self.batch_size]
                batch_tags = dev_tag_lists[ind: ind+self.batch_size]
                tensorized_sents, lengths = tensorized(batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                score = self.model(tensorized_sents, lengths)

                # Loss
                loss = self.cal_loss_func(
                    score, targets, tag2id
                ).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss
            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagid = self.best_model.test(
                tensorized_sents, lengths, tag2id
            )

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagid):
            tag_list = []
            for j in range(lengths[i]):
                tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 若indices = [1, 2, 0]则说明原来索引为1的元素映射到的新的索引是0
        # 索引是2的元素映射到新的索引是1
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists


def extend_maps(word2id, tag2id, for_crf=False):
    # LSTM模型训练的时候需要在word2id和tag2id中加入PAD和UNK
    # 如果是加入了CRF的lstm还需要加入<start>和<end>(解码时需要用到)
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def main():
    word2id, tag2id, train_word, train_tag = load_data("train")
    test_word, test_tag = load_data("test", make_vocab=False)
    dev_word, dev_tag = load_data("dev", make_vocab=False)

    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, False)
    # 接下来就是构造这个BiLSTM模型是如何训练的了

    model = load_model("BiLSTM.pkl")
    a = "文浩斌，男，在西安电子科技大学做汇报，一个平平无奇的干饭人罢了，也可能是个西安的汉族硕士研究生"
    sentence = [[i for i in a]]
    tensorized_sents, lengths = tensorized(sentence, word2id)
    tensorized_sents = tensorized_sents.to(model.device)
    with torch.no_grad():
        batch_tagid = model.best_model.test(
            tensorized_sents, [len(a)], bilstm_tag2id
        )
    pred_tag_lists = []
    id2tag = dict((id_, tag) for tag, id_ in bilstm_tag2id.items())
    for i, ids in enumerate(batch_tagid):
        tag_list = []
        for j in range(lengths[i]):
            tag_list.append(id2tag[ids[j].item()])
        pred_tag_lists.append(tag_list)
    for i in range(len(a)):
        print("{:>3s} --- {:>3s}".format(a[i], pred_tag_lists[0][i]))
    exit()

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)

    bilstm_model = BILSTMModel(vocab_size, out_size, False)
    bilstm_model.train(train_word, train_tag, dev_word, dev_tag,
                       bilstm_word2id, bilstm_tag2id)
    model_name = "BiLSTM"
    save_model(bilstm_model,  model_name + ".pkl")
    print("训练完毕，用时{}秒".format(int(time.time() - start)))

    print("\n\n评估模型...")
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word, test_tag, bilstm_word2id, bilstm_tag2id
    )

    metrics = Metrics(test_tag_lists, pred_tag_lists, True)
    metrics.report_scores()


if __name__ == '__main__':
    main()


import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from HMM import load_data
from BiLSTM import extend_maps
from copy import deepcopy
from utils import save_model
from evaluation import Metrics
from itertools import zip_longest
from BiLSTM import BiLSTM, sort_by_lengths, tensorized


def indexed(targets, tagset_size, start_id):
    # 将targets中的数转化为在[T * T]大小序列中的索引，T是标注的种类
    batch_size, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets


def cal_lstm_crf_loss(crf_scores, targets, tag2id):
    pad_id = tag2id.get("<pad>")
    start_id = tag2id.get("<start>")
    end_id = tag2id.get("<end>")

    device = crf_scores.device

    # [B, L] from [B, L, T, T]
    batch_size, max_len = targets.size()
    target_size = len(tag2id)

    # mask = 1 - ((targets == pad_id) + (targets == end_id))
    mask = (targets != pad_id)
    lengths = mask.sum(dim=1)
    targets = indexed(targets, target_size, start_id)

    # 计算Golden scores方法
    targets = targets.masked_select(mask)

    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size*target_size).contiguous()

    golden_scores = flatten_scores.gather(
        dim=1, index=targets.unsqueeze(1)
    ).sum()
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        # 当前时刻 有效的batch_size（因为有些序列比较短)
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                           t, start_id, :]
        else:
            # We add scores at current timestep to scores accumulated up to previous
            # timestep, and log-sum-exp Remember, the cur_tag of the previous
            # timestep is the prev_tag of this timestep
            # So, broadcast prev. timestep's cur_tag scores
            # along cur. timestep's cur_tag dimension
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()

    # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
    loss = (all_path_scores - golden_scores) / batch_size
    return loss


def preprocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        # 首先初始化父类中的变量
        super(BiLSTM_CRF, self).__init__()

        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)

        # CRF就是多学习一个转移矩阵，初始化为均匀分布
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size
        )

    def forward(self, sents_tensor, lengths):
        # 这里forward了一下
        emission = self.bilstm(sents_tensor, lengths)

        batch_size, max_len, out_size = emission.size()
        # 为什么把这个扩成 batch_size, max_len, out_size, out_size
        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1)+self.transition.unsqueeze(0)

        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        # 使用维特比算法进行解码
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tag_set_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device

        # B: batch_size; L: max_len; T: tag_size
        B, L, T, _ = crf_scores.size()

        viterbi = torch.zeros(B, L, T).to(device)
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)

        lengths = torch.LongTensor(lengths).to(device)

        # 前向递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                viterbi[:batch_size_t, step, :] = crf_scores[:batch_size_t, step, start_id, :]
                backpointer[:batch_size_t, step, :] = start_id
            else:
                # 这里的加是什么操作
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 开始回溯
        backpointer = backpointer.view(B, -1)
        tag_ids = []
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tag_set_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)
                ).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )
                index = torch.ones(batch_size_t).long() * (step * tag_set_size)
                index = index.to(device)
                index += offset.long()

            tags_t = backpointer[:batch_size_t].gather(
                dim=1, index=index.unsqueeze(1).long()
            )
            tags_t = tags_t.squeeze(1)
            tag_ids.append(tags_t.tolist())

        tag_ids = list(zip_longest(*reversed(tag_ids), fillvalue=pad))
        tag_ids = torch.Tensor(tag_ids).long()

        return tag_ids


class BILSTMModel(object):

    def __init__(self, vocab_size, out_size, crf=True):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.emb_size = 128
        self.hidden_size = 128
        self.crf = crf
        self.model = BiLSTM_CRF(vocab_size, self.emb_size, self.hidden_size, out_size).to(self.device)
        self.cal_loss_func = cal_lstm_crf_loss

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
        # 用作测试
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
            for j in range(lengths[i] - 1):
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


def main():
    word2id, tag2id, train_word, train_tag = load_data("train")
    test_word, test_tag = load_data("test", make_vocab=False)
    dev_word, dev_tag = load_data("dev", make_vocab=False)

    # 加了CRF的LSTM还要加入<start>和<end>
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, True)
    train_word, train_tag = preprocess_data_for_lstmcrf(train_word, train_tag)
    dev_word, dev_tag = preprocess_data_for_lstmcrf(dev_word, dev_tag)
    test_word, test_tag = preprocess_data_for_lstmcrf(test_word, test_tag, True)

    vocab_size = len(word2id)
    out_size = len(tag2id)

    model = BILSTMModel(vocab_size, out_size, True)
    model.train(train_word, train_tag, dev_word, dev_tag, crf_word2id, crf_tag2id)

    pred_tag_lists, test_tag_lists = model.test(test_word, test_tag, crf_word2id, crf_tag2id)
    metrics = Metrics(test_tag_lists, pred_tag_lists, True)
    metrics.report_scores()


if __name__ == '__main__':
    main()

import torch
from evaluation import Metrics


class HMM:
    def __init__(self, n_state, m_observe):
        self.n_state = n_state
        self.m_observe = m_observe
        self.A = torch.zeros(n_state, n_state)
        self.B = torch.zeros(n_state, m_observe)
        self.pi = torch.zeros(n_state)

    def train(self, word_lists, tag_lists, word2id, tag2id):
        # 按理来说有无监督的方法，但是囿于没看EM算法，先用统计的方法实现
        assert len(word_lists) == len(tag_lists)

        # 只能使用统计的方法
        for tag_list in tag_lists:
            seq_length = len(tag_list)
            for i in range(seq_length - 1):
                current_tag_id = tag2id[tag_list[i]]
                next_tag_id = tag2id[tag_list[i+1]]
                self.A[current_tag_id][next_tag_id] += 1

        self.A[self.A == 0.] = 1e-10
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        for tag_list, word_list in zip(tag_lists, word_lists):
            seq_length = len(tag_list)
            for i in range(seq_length):
                current_tag_id = tag2id[tag_list[i]]
                current_word_id = word2id[word_list[i]]
                self.B[current_tag_id][current_word_id] += 1

        self.B[self.B == 0.] = 1e-10
        self.B = self.B/self.B.sum(dim=1, keepdim=True)

        for tag_list in tag_lists:
            init_tagid = tag2id[tag_list[0]]
            self.pi[init_tagid] += 1

        self.pi[self.pi == 0.] = 1e-10
        self.pi = self.pi / self.pi.sum()

    def test(self, word_lists, word2id, tag2id):
        pred_tag_lists = []
        for word_list in word_lists:
            pred_tag_list = self.decode(word_list, word2id, tag2id)
            pred_tag_lists.append(pred_tag_list)
        return pred_tag_lists

    def decode_1(self, word_list, word2id, tag2id):
        T = len(word_list)
        # 前向计算
        # 结果：各个时段φ(i)的值
        # 初始化

        # 问题：当链条很长的时候，十分多的小概率相乘，可能会造成下溢
        # 解决方案：将转移矩阵映射到log函数空间上，这样乘的运算还能化为加运算！
        # 来实现一下映射的过程
        A = torch.log(self.A)
        B = torch.log(self.B)
        pi = torch.log(self.pi)

        theta = torch.zeros(self.n_state)
        phi = torch.zeros(T, self.n_state)
        size_of_state = self.n_state
        for i in range(size_of_state):
            # theta[i] = self.pi[i] * self.B[i][word2id[word_list[0]]]
            if word_list[0] not in word2id.keys():
                theta[i] = pi[i] + 1/self.n_state
            else:
                theta[i] = pi[i] + B[i][word2id[word_list[0]]]

        for i in range(1, T):
            for ii in range(size_of_state):
                max_theta = 0
                max_phi = 0
                max_idx = 0
                for iii in range(size_of_state):
                    # phi = theta[iii] * self.A[iii][ii]
                    phi_ = theta[iii] + A[iii][ii]
                    if phi_ > max_phi:
                        max_phi = phi_
                        max_idx = iii
                    # theta_ = phi * self.B[ii][word2id[word_list[i]]]
                    if word_list[i] not in word2id.keys():
                        theta_ = phi_ + 1/self.n_state
                    else:
                        theta_ = phi_ + B[ii][word2id[word_list[i]]]
                    if theta_ > max_theta:
                        max_theta = theta_

                theta[ii] = max_theta
                phi[i][ii] = max_idx

        # 反向回溯
        # 结果：从第一个词开始到最后一个词对应的状态序列
        state = torch.zeros(T)
        state[-1] = theta.argmax()
        for i in range(T-2, -1, -1):
            # 调试第二处，取的值为Tensor形式，而需要为long, byte or bool形式，而且索引需要使用int而不能是Tensor
            state[i] = int(phi[i+1][int(state[i+1])])

        assert len(state) == len(word_list)
        # 修改3，items而不是item
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tag_list = [id2tag[int(i)] for i in state]
        return tag_list

    def decode(self, word_list, word2id, tag2id):
        # 采用对数概率，这样源空间中很小的概率，被映射到对数空间的大的负数
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.pi)

        # 初始化维特比矩阵 维度为[状态数, 序列长度]
        seq_len = len(word_list)
        viterbi = torch.zeros(self.n_state, seq_len)

        # back_pointer 大小与维特比矩阵相同
        # back_pointer相当于φ，用于进行回溯
        back_pointer = torch.zeros(self.n_state, seq_len).long()

        # Pi[i]表示第一个字标记为i的概率
        # Bt[word_id]表示字为word_id时，对应各个标记的概率
        # self.B.t()[tag_id]表示各个状态转移到tag对应的概率

        start_wordid = word2id.get(word_list[0], None)
        Bt = B.t()
        if start_wordid is None:
            # 如果字不在字典里，则假设状态的概率分布是均匀的
            bt = torch.log(torch.ones(self.n_state) / self.n_state)
        else:
            bt = Bt[start_wordid]

        # 在进行第一步的初始化
        viterbi[:, 0] = Pi + bt
        # 但是这里用于回溯的矩阵将第一列初始化为-1
        back_pointer[:, 0] = -1

        # 开始前向递推
        for step in range(1, seq_len):
            word_id = word2id.get(word_list[step], None)
            # 处理字不在字典中的情况
            if word_id is None:
                bt = torch.log(torch.ones(self.n_state) / self.n_state)
            else:
                bt = Bt[word_id]
            for tag_id in range(len(tag2id)):
                max_prob, max_id = torch.max(
                    viterbi[:, step-1] + A[:, tag_id],
                    dim=0
                )
                viterbi[tag_id, step] = max_prob + bt[tag_id]
                back_pointer[tag_id, step] = max_id

        # 终止
        best_path_prob, best_path_pointer = torch.max(
            viterbi[:, seq_len-1],
            dim=0
        )

        # 回溯
        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len-1, 0, -1):
            best_path_pointer = back_pointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)
        assert len(best_path) == len(word_list)

        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]
        return tag_list


def load_data(split, make_vocab=True):
    assert split in ['train', 'test', 'dev']

    word_lists = []
    tag_lists = []
    with open(split + ".char.bmes", 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 若make_vocab为True，还需要返回word2id，tag2id
    if make_vocab is True:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word2id, tag2id, word_lists, tag_lists
    return word_lists, tag_lists


def build_map(target_lists):
    maps = {}
    for i in target_lists:
        for ii in i:
            if ii not in maps.keys():
                maps[ii] = len(maps)
    return maps


def main():
    train_word2id, train_tag2id, train_word, train_tag = load_data('train', make_vocab=True)
    test_word, test_tag = load_data('test', make_vocab=False)
    dev_word, dev_tag = load_data('dev', make_vocab=False)

    model = HMM(len(train_tag2id), len(train_word2id))
    model.train(train_word, train_tag, train_word2id, train_tag2id)
    pred_sequence = model.test(test_word, train_word2id, train_tag2id)
    metrics = Metrics(test_tag, pred_sequence, False)
    metrics.report_scores()
    metrics.test(model, train_word2id, train_tag2id)


if __name__ == '__main__':
    main()

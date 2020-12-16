from collections import Counter
from utils import flatten_lists


class Metrics(object):

    def __init__(self, golden_tags, predict_tags, remove_o=False):
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        # 我感觉去掉O是个没啥必要的行为，就是为了校验罢了，还是加上吧
        if remove_o:
            self.remove_otags()

        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tag()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

        # 计算精确率
        self.precision_scores = self.cal_precision()
        self.recall_scores = self.cal_recall()
        self.f1 = self.cal_f1()

    def remove_otags(self):
        o_indx = [i for i in range(len(self.golden_tags)) if self.golden_tags[i] == "O"]
        self.golden_tags = [value for i, value in enumerate(self.golden_tags) if i not in o_indx]
        self.predict_tags = [value for i, value in enumerate(self.predict_tags) if i not in o_indx]

    def count_correct_tag(self):
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1
        return correct_dict

    def cal_precision(self):
        precision_scores = {}
        for tag in self.tagset:
            precision_scores[tag] = self.correct_tags_number.get(tag, 0) / self.predict_tags_counter[tag]

        return precision_scores

    def cal_recall(self):
        recall_score = {}
        for tag in self.tagset:
            recall_score[tag] = self.correct_tags_number.get(tag, 0) / self.golden_tags_counter[tag]

        return recall_score

    def cal_f1(self):
        f1_score = {}
        for tag in self.tagset:
            f1_score[tag] = 2 * (self.precision_scores[tag] * self.recall_scores[tag]) / \
                                (self.precision_scores[tag] + self.recall_scores[tag] + 1e-10)
        return f1_score

    def report_scores(self):
        header_format = "{:>9s}  {:>9} {:>9} {:>9} {:>9}"
        header = ["precision", "recall", "f1-scores", "support"]
        # 这里有个star expression，在format中意思是依次取值
        print(header_format.format("", *header))

        row_format = "{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}"
        for tag in self.tagset:
            print(row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1[tag],
                self.golden_tags_counter[tag]
            ))

        weighted_average = self._cal_weighted_average()
        print(row_format.format(
            "avg/total",
            weighted_average['precision'],
            weighted_average['recall'],
            weighted_average['f1_score'],
            len(self.golden_tags)
        ))

    def _cal_weighted_average(self):
        weighted_average = {}
        total = len(self.golden_tags)

        weighted_average['precision'] = 0
        weighted_average['recall'] = 0
        weighted_average['f1_score'] = 0

        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def test(self, model, word2id, tag2id):
        sentence = "文浩斌，男，在西安电子科技大学做汇报，一个平平无奇的干饭人罢了，也可能是个西安的汉族硕士研究生"
        pred_sequence_ = model.test([[i for i in sentence]], word2id, tag2id)
        for i in range(len(pred_sequence_[0])):
            print(sentence[i] + "---" + pred_sequence_[0][i])

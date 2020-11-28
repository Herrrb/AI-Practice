from sklearn_crfsuite import CRF
from HMM import load_data
from evaluation import Metrics
from sklearn_crfsuite.scorers import make_scorer, metrics
from scipy.stats import expon
from sklearn.model_selection import RandomizedSearchCV


def word2features(sent, i):
    # 所以这里的features是由CRF这个别人写好的东西提前规定的
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i-1]
    next_word = "</s>" if i == (len(sent) - 1) else sent[i+1]
    features = {
        "w": word,
        "w-1": prev_word,
        "w+1": next_word,
        "w-1:w": prev_word+word,
        "w:w+1": word+next_word,
        "bias": 1
    }
    return features


def flatten_list(list_):
    result = []
    for i in list_:
        for ii in i:
            result.append(ii)
    return result


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


class CRFModel(object):
    def __init__(self, algorithm='lbfgs', c1=0.72, c2=0.25,
                 max_iterations=100, all_posible_transitions=False):
        self.model = CRF(algorithm=algorithm, c1=c1, c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_posible_transitions)

    def train(self, sentence, tag_list):
        features = [sent2features(s) for s in sentence]
        # 这里需要把这个库tutorial好好看看了
        self.model.fit(features, tag_list)
        return self.model.classes_

    def test(self, sentences):
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists


def main():
    train_word, train_tag = load_data('train', make_vocab=False)
    test_word, test_tag = load_data('test', make_vocab=False)
    crf_model = CRFModel()
    labels = crf_model.train(train_word, train_tag)
    pred_sequence = crf_model.test(test_word)
    metrics_ = Metrics(test_tag, pred_sequence, False)
    metrics_.report_scores()

    # 优化参数
    # crf_model = CRF(algorithm='lbfgs',
    #                 max_iterations=100,
    #                 all_possible_transitions=True)
    #
    # params_space = {
    #     'c1': expon(0.5),
    #     'c2': expon(0.05),
    # }
    #
    # f1_score = make_scorer(metrics.flat_f1_score,
    #                        average='weighted',
    #                        labels=labels)
    #
    # rs = RandomizedSearchCV(
    #     crf_model, params_space,
    #     cv=3, verbose=1, n_jobs=1, n_iter=50, scoring=f1_score
    # )

    # train_word = flatten_list(train_word)
    # train_tag = flatten_list(train_tag)

    # features = [sent2features(s) for s in train_word]
    #
    # rs.fit(features, train_tag)
    #
    # print("best params: ", rs.best_params_)
    # print("best cv score: ", rs.best_score_)
    # print("model size: {:0.2f}M".format(rs.best_estimator_._size_ / 1000000))


if __name__ == '__main__':
    main()

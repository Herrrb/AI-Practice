import pickle


def save_model(model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_name):
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model


def flatten_lists(lists):
    result = []
    for item in lists:
        if type(item) is list:
            for i in item:
                result.append(i)
        else:
            result.append(item)

    return result

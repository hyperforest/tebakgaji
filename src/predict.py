import json
import pickle

import numpy as np
import pandas as pd

'''
Sample format:

sample = [
    {
        'role': 'data scientist',
        'company': 'gojek',
        'years_of_exp': 3,
        'city': 'jakarta'
    },
    ...
]

(see `src/sample.json`)
'''

countvec_dir = './model/count_vectorizer.pkl'
catboost_model_dir = './model/catboost_model.pkl'

with open(countvec_dir, 'rb') as file:
    count_vectorizer = pickle.load(file)

with open(catboost_model_dir, 'rb') as file:
    catboost_model = pickle.load(file)


def predict(data: pd.DataFrame):
    counts = count_vectorizer.transform(data.role).toarray().tolist()
    X = np.hstack([counts, data.drop('role', axis=1).values])

    y_pred = catboost_model.predict(X).tolist()
    return y_pred


def main():
    with open('./datasets/sample.json', 'r') as file:
        sample = json.load(file)

    data = pd.DataFrame(sample['data'])
    predictions = predict(data)

    for i in range(len(data)):
        print('Data:')
        print(json.dumps(data.iloc[i].to_dict(), indent=4))
        print('Predicted salary: %.1fM' % predictions[i])


if __name__ == '__main__':
    main()

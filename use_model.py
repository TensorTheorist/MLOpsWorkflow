import pickle
import numpy as np

local_classifier = pickle.load(open('pickle/classifier.pickle', 'rb'))
local_scaler = pickle.load(open('pickle/scaler.pickle', 'rb'))


def predictWithProb(age, salary):
    new_pred = local_classifier.predict(local_scaler.transform(np.array([[age, salary]])))
    new_prob = local_classifier.predict_proba(local_scaler.transform(np.array([[age, salary]])))

    return new_pred, new_prob[:, 1]


print(predictWithProb(20, 50000))

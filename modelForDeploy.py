import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

training_data = pd.read_csv('StorePurchaseData.csv')

X = training_data.iloc[:, :-1].values
y = training_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)

print(y_pred)
print(y_prob)

print(confusion_matrix(y_test, y_pred))

model_file = "pickle/classifier.pickle"
pickle.dump(classifier, open(model_file, 'wb'))

scaler_file = "pickle/scaler.pickle"
pickle.dump(sc, open(scaler_file, 'wb'))



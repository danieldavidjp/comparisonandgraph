from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import pandas as pd
import pickle

data = pd.read_csv('datos.csv')
labels = pd.read_csv('labels.csv')
X = np.array(data)
y = np.ravel(labels)
clf = QuadraticDiscriminantAnalysis()
clf.fit(X, y)
#print(clf.predict([dato]))

pkl_filename ="pickle_model.pkl"

with open(pkl_filename, 'wb') as file:
    pickle.dump(clf,file)


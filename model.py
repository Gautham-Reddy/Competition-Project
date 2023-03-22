from sklearn import svm
import numpy as np
import os
import re

folder_path = "/Users/gauthamreddy/Desktop/Competition Project/data/TrainingData"
session_set = set()

for filename in os.listdir(folder_path):
    match = re.search(r'subject_(\d+)_(\d+)__(\w+)\.csv', filename)
    
    if match:
        session_set.add((match.group(1), match.group(2)))

X, Y = [], []

for session in session_set:
    # Load CSV file into numpy array
    x_data = np.genfromtxt(folder_path + '/subject_{}_{}__{}.csv'.format(session[0], session[1], 'x'), delimiter=',')
    x_time = np.genfromtxt(folder_path + '/subject_{}_{}__{}.csv'.format(session[0], session[1], 'x_time'), delimiter=',')
    y_data = np.genfromtxt(folder_path + '/subject_{}_{}__{}.csv'.format(session[0], session[1], 'y'), delimiter=',')
    y_time = np.genfromtxt(folder_path + '/subject_{}_{}__{}.csv'.format(session[0], session[1], 'y_time'), delimiter=',')

    if  x_data[1::4].shape[0] != y_data.shape[0]: print("False: " + x_data[1::4].shape[0] + ", " + y_data.shape[0])
    # Print the shape and contents of the numpy array
    X += x_data[1::4].tolist()
    Y += y_data.reshape((y_data.shape[0], 1)).tolist()

print(len(X))
print(len(Y))

X = np.array(X)
Y = np.array(Y)
X_training = X[:int(len(X) * 0.1)]
X_test = X[int(len(X) * 0.1):]
Y_training = Y[:int(len(Y) * 0.1)]
Y_test = Y[int(len(Y) * 0.1):]

clf = svm.SVC()
print("starting training")
clf.fit(X_training, Y_training)
print("ending training")

num_correct = 0

for i, row in enumerate(X_test):
    if clf.predict([row]) == Y_test[i]: num_correct += 1
    print(num_correct / (i + 1))


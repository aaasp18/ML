#!/usr/bin/python3

import numpy
import time
import csv
import logging
import gzip
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import train_test_split


logger = logging.getLogger('learn')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('learn.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - [%(name)s/%(funcName)s] - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

algorithms = {}

names = [
  "Nearest Neighbors",
  "Linear SVM",
  # "RBF SVM",
  # "Gaussian Process",
  "Decision Tree",
  "Random Forest",
  "Neural Net",
  "AdaBoost",
  "Naive Bayes",
  # "QDA"
]

classifiers = [
  KNeighborsClassifier(3),
  SVC(kernel="linear", C=0.025, probability=True),
  # SVC(gamma=2, C=1, probability=True),
  # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
  DecisionTreeClassifier(),
  RandomForestClassifier(n_estimators=200),
  MLPClassifier(alpha=0.001),
  AdaBoostClassifier(),
  GaussianNB(),
  # QuadraticDiscriminantAnalysis()
]

# names = [
#   "Random Forest"
# ]

# classifiers = [
#   RandomForestClassifier(n_estimators=200)
# ]

def save(algorithms, save_file):
  t = time.time()
  f = gzip.open(save_file, 'wb')
  pickle.dump(algorithms, f)
  f.close()
  logger.debug("{:d} ms".format(int(1000 * (t - time.time()))))

def learn(fname):
  t = time.time()
  header = []
  rows = []
  with open(fname, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
      if i == 0:
        header = row
      else:
        for j, val in enumerate(row):
          if val == '':
            row[j] = 0
            continue
          try:
            if(j != 0):
              row[j] = float(val)
          except:
            logger.debug("exception in row {}, column {}".format(i,j))
            row[j] = 0
            continue
        rows.append(row)

  y = numpy.empty(len(rows), dtype="<U30")
  x = numpy.zeros((len(rows), len(rows[0]) - 1))

  record_range = list(range(len(rows)))
  for i in record_range:
    y[i] = rows[i][0]
    x[i, :] = numpy.array(rows[i][1:])

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle = True, random_state=24)

  for name, clf in zip(names, classifiers):
    t2 = time.time()
    logger.debug("learning {}".format(name))
    try:
        algorithms[name] = clf.fit(X_train, y_train)
        logger.debug("learned {}, {:d} ms".format(
            name, int(1000 * (t2 - time.time()))))
        score = algorithms[name].score(X_test,y_test)
        logger.debug("{}, score: {}".format(name,score))
    except Exception as e:
        logger.error("{} {}".format(name, str(e)))
    print('\n')

learn("./data.csv")
save(algorithms,"laberinto.ai")
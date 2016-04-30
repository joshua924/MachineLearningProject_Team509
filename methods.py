import numpy as np
import feature_generation as fg

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def run_random_forest(X, y, n_tr, n_te):
  X_tr, y_tr = X[:n_tr], y[:n_tr]
  print 'Training Random Forest Model ...'
  model = RandomForestClassifier(n_estimators=20, bootstrap=True, 
      n_jobs=-1, max_features='auto').fit(X_tr, y_tr)
  print 'Evaluating ...'
  print 'Score on training dataset is %6.4f' \
      % model.score(X_tr, y_tr)
  print 'Score on test dataset is %6.4f' % \
      model.score(X[-n_te:], y[-n_te:])


# def get_adaBoost_model(X, y, n_round):
#   trees = []
#   alphas = []
#   weight = np.ones(y.shape[0])
#   weight /= np.linalg.norm(weight)
#   for i in range(n_round):
#     clf = DecisionTreeClassifier(max_depth=5).fit(X, y, sample_weight=weight)
#     trees.append(clf)
#     score = clf.score(X, y)
#     alpha = np.log(score / (1-score))
#     alphas.append(alpha)
#     predicted = clf.predict(X)
#     for i in range(y.shape[0]):
#       if predicted[i] != y[i]:
#         weight[i] *= score / (1-score)
#   return trees, alphas
#
# def run_adaBoost(X, y, n_tr, n_te, n_round=10):
#   print 'Training boosting model ...'
#   trees, alphas = get_adaBoost_model(X[:n_tr], y[:n_tr], n_round)
#   print 'Evaluating ...'
#   s = n_tr
#   vals = np.zeros(s)
#   for i, tree in enumerate(trees):
#     predicted = np.array(map(lambda x:-1 if x=='dire' else 1, tree.predict(X[:n_tr])))
#     vals += predicted * alphas[i]
#   hit = np.array(map(lambda x:'dire' if x < 0 else 'radiant', vals))
#   print 'Score on training dataset is %6.4f' % (float(np.sum(hit == y[:n_tr])) / s)
#   s = n_te
#   vals = np.zeros(s)
#   for i, tree in enumerate(trees):
#     predicted = np.array(map(lambda x:-1 if x=='dire' else 1, tree.predict(X[-n_te:])))
#     vals += predicted * alphas[i]
#   hit = np.array(map(lambda x:'dire' if x < 0 else 'radiant', vals))
#   print 'Score on training dataset is %6.4f' % (float(np.sum(hit == y[-n_te:])) / s)


def run_logistic(X, y, n_tr, n_te):
  X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[-n_te:], y[-n_te:]

  print 'Training Logistic Model ...'
  model = LogisticRegression(penalty='l2', n_jobs=4, tol=0.0000001) \
      .fit(X_tr, y_tr)

  print 'Evaluating ...'
  print 'Training Accuracy is %6.4f' % model.score(X_tr, y_tr)
  print 'Test Accuracy is %6.4f' % model.score(X_te, y_te)


def fake_generate_features():
  y = np.loadtxt('feature_selected/y.txt', dtype='S8')
  list_file = ['feature_selected/X_10_0.02.txt', 'feature_selected/X_10_0.015.txt', 
      'feature_selected/X_20_0.01.txt', 'feature_selected/X_20_0.02.txt', 
      'feature_selected/X_20_0.015.txt']
  for file in list_file:
    print 'Using file ', file
    X = np.loadtxt(file, delimiter=',', dtype='int', skiprows=1)
    run_logistic(X, y, 10000, 2000)
    run_random_forest(X, y, 10000, 2000)
    run_adaBoost(X, y, 10000, 2000, 11)
    print '\n'

if __name__ == '__main__':
  print 'Generating Features ...'
  #X, y = fg.generate_feature('match_player.csv', min_count=100, single_only=True)
  X, y = fake_generate_features()
  # run_logistic(X, y, 10000, 2000)
  # run_random_forest(X, y, 10000, 2000)
  # run_adaBoost(X, y, 10000, 2000, 11)


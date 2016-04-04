import numpy as np
import feature_generation as fg

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def run_random_forest(X, y, n_tr, n_te):
  X_tr, y_tr = X[:n_tr], y[:n_tr]
  print 'Training Model ...'
  model = RandomForestClassifier(n_estimators=20, bootstrap=True, 
      n_jobs=-1, max_features='auto').fit(X_tr, y_tr)
  print 'Evaluating ...'
  print 'Score on training dataset is %6.4f' \
      % model.score(X_tr, y_tr)
  print 'Score on test dataset is %6.4f' % \
      model.score(X[-n_te:], y[-n_te:])


def get_adaBoost_model(X, y, n_round=10):
  trees = []
  alphas = []
  weight = np.ones(y.shape[0])
  weight /= np.linalg.norm(weight)
  for i in range(n_round):
    clf = DecisionTreeClassifier(max_depth=5).fit(X, y, sample_weight=weight)
    trees.append(clf)
    score = clf.score(X, y)
    alpha = np.log(score / (1-score))
    alphas.append(alpha)
    predicted = clf.predict(X)
    for i in range(y.shape[0]):
      if predicted[i] != y[i]:
        weight[i] *= score / (1-score)
  return trees, alphas


def adaBoost_eval(trees, alphas, X, y):
  s = y.shape[0]
  vals = np.zeros(s)
  for i, tree in enumerate(trees):
    predicted = np.array(map(lambda x:-1 if x=='dire' else 1, tree.predict(X)))
    vals += predicted * alphas[i]
  hit = np.array(map(lambda x:'dire' if x < 0 else 'radiant', vals))
  return float(np.sum(hit == y)) / s


'''
0.9986/0.5895 for log loss, l1 penalty and 20 iterations
0.9693/0.5925 for log loss, l2 penalty and 20 iterations
0.9760/0.5880 for log loss, l1 penalty and 30 iterations
0.9934/0.5890 for hinge loss, l2 penalty and 20 iterations
1.0000/0.5840 for modified_huber loss, l2 penalty and 20 iterations
'''
def run_logistic(X, y, n_tr, n_te):
  X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[-n_te:], y[-n_te:]

  print 'Training Model ...'
  model = LogisticRegression(penalty='l2', n_jobs=4, tol=0.0000001) \
      .fit(X_tr, y_tr)

  print 'Evaluating ...'
  print 'Training Error is %6.4f' % model.score(X_tr, y_tr)
  print 'Raw Precision is %6.4f' % model.score(X_te, y_te)


if __name__ == '__main__':
  print 'Generating Features ...'
  X, y = fg.generate_feature('match_player.csv')
  # run_logistic(X, y, 10000, 2000)
  run_random_forest(X, y, 10000, 2000)


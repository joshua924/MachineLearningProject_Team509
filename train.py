import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso, SGDClassifier


def run_random_forest(X, y, n_tr, n_te):
  X_tr, y_tr = X[:n_tr], y[:n_tr]
  print 'Training Random Forest Model ...'
  model = RandomForestClassifier(n_estimators=20, bootstrap=True, 
      n_jobs=-1, max_features='auto').fit(X_tr, y_tr)
  print 'Evaluating ...'
  print 'Score on training dataset is %6.4f' \
      % model.score(X_tr, y_tr)
  print 'Score on validation dataset is %6.4f' % \
      model.score(X[-n_te:], y[-n_te:])


def run_logistic(X, y, n_tr, n_te, save=False):
  X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[-n_te:], y[-n_te:]
  model = LogisticRegression(penalty='l2', n_jobs=4, tol=0.0000001) \
      .fit(X_tr, y_tr)
  print 'Training, validation accuracy is %6.4f and %6.4f' % \
      (model.score(X_tr, y_tr), model.score(X_te, y_te))
  if save:
    pickle.dump(model, open('model/log_reg.model', 'w'))


def run_SGD(X, y, n_tr, n_te):
  X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[-n_te:], y[-n_te:]
  penalties = ['hinge', 'log']
  for p in penalties:
    model = SGDClassifier(loss=p, penalty=None, n_iter=100).fit(X_tr, y_tr)
    print 'Training, validation accuracy is %6.4f and %6.4f for %s loss' % \
        (model.score(X_tr, y_tr), model.score(X_te, y_te), p)


def get_feature():
  X = np.loadtxt('feature_selected/selectedX.txt', delimiter=',')
  y = np.loadtxt('feature_selected/y.txt', delimiter=',', dtype='int')
  return X, y


if __name__ == '__main__':
  print 'Generating and Selecting Features ...'
  X, y = get_feature()
  print 'Building Model ...'
  # run_svc(X, y, 10000, 2000)
  run_logistic(X, y, 10000, 2000, save=True)
  # run_SGD(X, y, 10000, 2000)


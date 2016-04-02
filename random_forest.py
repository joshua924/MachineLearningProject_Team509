import numpy as np
import feature_generation as fg

from sklearn.ensemble import RandomForestClassifier

def run_random_forest(raw_input_file, n_training, n_test):
  print 'Generating Features ...'
  X, y = fg.generate_feature(raw_input_file)
  X_training, y_training = X[:n_training], y[:n_training]
  X_test, y_test = X[-n_test:], y[-n_test:]

  print 'Building random forest ...'
  model = RandomForestClassifier(n_estimators=10, bootstrap=True, 
      n_jobs=-1, max_features='auto').fit(X_training, y_training)

  print 'Running evaluation ...'
  print 'Score on test set is %5.3f' % model.score(X_test, y_test)

run_random_forest('match_player.csv', 10000, 2000)
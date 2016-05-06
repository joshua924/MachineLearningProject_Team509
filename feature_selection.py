import numpy as np
import feature_generation as fg


def feature_selection(dump=False):
  feature_dict, X, y = fg.generate_feature('match_player.csv')
  y = np.array([-1 if t == 'dire' else 1 for t in y])
  newY = y.reshape([y.shape[0], 1])
  corrs = np.corrcoef(np.hstack((X, newY)), rowvar=0)[-1]
  newX = np.delete(X, np.where(corrs<0.011), 1)
  if dump:
    f = feature_dict.items()
    f.sort(key=lambda x : x[1])
    f = np.array([t[0] for t in f])
    f = np.delete(f, np.where(corrs<0.011))
    np.savetxt('preprocessing/selected_features.txt', f, fmt='%s', delimiter=',')
  return newX, y


if __name__ == '__main__':
  X, y = feature_selection(True)
  np.savetxt('feature_selected/selectedX.txt', X, fmt='%s', delimiter=',')
  np.savetxt('feature_selected/y.txt', y, fmt='%s', delimiter=',')


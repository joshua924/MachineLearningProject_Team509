'''
the feature set includes:
1. Each hero's appearance
2. Each combination of two heroes on Dire side
3. Each combination of two heroes on Radiant side
4. Each combination of two heroes on different sides
'''
import numpy as np

'''
@param in_file: the raw input file
return (X, y): X is a sparse matrix containing features, 
and y is a list of labels for each instance
'''
def generate_feature(in_file, dump=False):
  f = open(in_file, 'r')
  f.readline()
  training_data, tags = [], []
  total_features = {}

  for line in f.readlines():
    tokens = line.replace('\n', '').split(',')
    fs = [s for s in tokens[1:] if s.isdigit()]
    # ignore invalid data
    if len(fs) != 10:
      continue
    tags.append(tokens[0])
    features = get_feature_array(fs)
    update_total_features(total_features, features)
    training_data.append(features)

  training_data = transform_to_matrix(total_features, training_data)
  if dump == True:
    np.savetxt('preprocessing/dump.txt', training_data, fmt='%d', delimiter=',')
  tags = np.array(tags)
  return training_data, tags


def get_feature_array(fs):
  features = []
  features.extend(make_single_feature(fs))
  features.extend(make_same_side_bi_feature('d', fs[:5]))
  features.extend(make_same_side_bi_feature('r', fs[5:]))
  features.extend(make_diff_side_bi_feature(fs))
  return features


def make_single_feature(fs):
  features = []
  for i in range(10):
    side = 'd' if i < 5 else 'r'
    features.append('{}_{}'.format(side, fs[i]))
  return features


def make_same_side_bi_feature(side, fs):
  if len(fs) != 5:
    return
  features = []
  for i in range(5):
    for j in range(i+1, 5):
      features.append('{}_{}_{}_{}'.format(side, fs[i], side, fs[j]))
  return features


def make_diff_side_bi_feature(fs):
  features = []
  for i in range(5):
    for j in range(5, 10):
      features.append('d_{}_r_{}'.format(fs[i], fs[j]))
  return features


def update_total_features(total_features, features):
  index = len(total_features)
  for each in features:
    if each not in total_features:
      total_features[each] = index
      index += 1


def transform_to_matrix(total_features, training_data):
  transformed = np.zeros((len(training_data), len(total_features)))
  for i, features in enumerate(training_data):
    for feature in features:
      transformed[i][total_features[feature]] = 1
  return transformed


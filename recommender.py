import pickle
import numpy as np


print 'Loading the model and selected features ...'
model = pickle.load(open('model/log_reg.model', 'r'))
features = np.loadtxt('feature_selected/selected_features.txt', delimiter=',', dtype='S')
print 'Done'

feature_dict = {}
for i, each in enumerate(features):
  feature_dict[each] = i
print feature_dict
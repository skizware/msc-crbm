import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import scale
from experiment import TimitExperiment
import os

DIR_COMPONENT_OUTPUT = '/home/dave/code/msc-crbm/test/v2/TIMIT/pre-processed/'
FILENAME_COMPONENT_OUTPUT = 'principle_components_model.pkl'

pca_freq = PCA(n_components=80, whiten=True)

data_refs = TimitExperiment.load_data_sets()[0]
data_arr = []

print "Loading data"
for ref in data_refs:
    specgram = np.load(ref)[0]
    data_arr.append(specgram.T)

data_arr = np.array(data_arr)

print "DATA LOAD COMPLETED - Shape = {}".format(data_arr.shape)

data_arr = data_arr.reshape(data_arr.shape[0] * data_arr.shape[1], data_arr.shape[2])
np.random.shuffle(data_arr)

# print "Scaling data:"
# data_arr = scale(data_arr, axis=0)

means = data_arr.mean(axis=0)
stds = data_arr.std(axis=0)

for count in xrange(0, means.shape[0]):
    print "mean[{}] = {}, std[{}] = {}".format(count, means[count], count, stds[count])

print "Attempting to fit data"
pca_freq.fit(data_arr)
print "Explained variance = {}".format(np.sum(pca_freq.explained_variance_))
print "Component shape = {}".format(pca_freq.components_.shape)

file_dir = DIR_COMPONENT_OUTPUT + FILENAME_COMPONENT_OUTPUT

if not os.path.exists(os.path.dirname(file_dir)):
    os.makedirs(os.path.dirname(file_dir))

joblib.dump(pca_freq, file_dir)

data_arr_transformed = pca_freq.transform(data_arr)

means = data_arr_transformed.mean(axis=0)
stds = data_arr_transformed.std(axis=0)

for count in xrange(0, means.shape[0]):
    print "mean[{}] = {}, std[{}] = {}".format(count, means[count], count, stds[count])

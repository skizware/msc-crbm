import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from experiment import TimitExperiment
import os

DIR_COMPONENT_OUTPUT='/home/dave/code/msc-crbm/test/v2/TIMIT/pre-processed/'
FILENAME_COMPONENT_OUTPUT='principle_components_model.pkl'

pca_freq = PCA(n_components=80, whiten=True)

data_refs = TimitExperiment.load_data_sets()[0]
data_arr = []

print "Loading data"
for ref in data_refs:
    specgram = np.load(ref)[0]
    data_arr.append(specgram.swapaxes(0,1))

data_arr = np.array(data_arr)

print "DATA LOAD COMPLETED - Shape = {}".format(data_arr.shape)

data_arr = data_arr.reshape(data_arr.shape[0]*data_arr.shape[1], data_arr.shape[2])
np.random.shuffle(data_arr)

print "Attempting to fit data"
pca_freq.fit(data_arr)
print "Explained variance = {}".format(np.sum(pca_freq.explained_variance_))
print "Component shape = {}".format(pca_freq.components_.shape)

file_dir = DIR_COMPONENT_OUTPUT + FILENAME_COMPONENT_OUTPUT

if not os.path.exists(os.path.dirname(file_dir)):
    os.makedirs(os.path.dirname(file_dir))

joblib.dump(pca_freq, file_dir)

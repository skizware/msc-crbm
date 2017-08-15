from subprocess import Popen, PIPE
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import os

DIR_TIMIT_TRAIN_SET = '/home/dave/code/msc-crbm/test/v2/TIMIT_TRAIN_NO_CHUNK/pre-processed'
DIR_PCA_MODEL_OUT = '/home/dave/code/msc-crbm/test/v2/PCA_MODEL/scaled/'
FILENAME_PCA_MODEL = 'pca_model.pkl'

proc = Popen(['find', DIR_TIMIT_TRAIN_SET, '-iname', '*.npy'], stdout=PIPE)
data_refs = proc.stdout.read().split('\n')
data_refs = data_refs[:len(data_refs) - 1]

myScaler = joblib.load('/home/dave/code/msc-crbm/test/v2/scaler/scaler_dump.pkl')

print "Loading data"
allData = tuple()
for ref in data_refs:
    specgram = np.load(ref)[0]
    allData += (specgram.copy().T,)

allData = np.concatenate(allData)

print allData.shape

print "SCALING DATA - START"
allDataScaled = myScaler.transform(allData)
print "SCALING DATA - END"

pca_model = PCA(n_components=80, whiten=True)

print "Fitting PCA model - START"
pca_model.fit(allDataScaled)
print "Fitting PCA model - END"

if not os.path.exists(DIR_PCA_MODEL_OUT):
    os.makedirs(DIR_PCA_MODEL_OUT)

joblib.dump(pca_model, DIR_PCA_MODEL_OUT + FILENAME_PCA_MODEL)

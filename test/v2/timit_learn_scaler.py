from subprocess import Popen, PIPE
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import os

SCALER_FILE_NAME = 'scaler_dump.pkl'

DIR_TIMIT_TRAIN_SET = '/home/dave/code/msc-crbm/test/v2/TIMIT_TRAIN_NO_CHUNK/pre-processed'
DIR_SCALER_OUT = '/home/dave/code/msc-crbm/test/v2/scaler/'

proc = Popen(['find', DIR_TIMIT_TRAIN_SET, '-iname', '*.npy'], stdout=PIPE)
data_refs = proc.stdout.read().split('\n')
data_refs = data_refs[:len(data_refs) - 1]

print "Loading data"
allData = tuple()
for ref in data_refs:
    specgram = np.load(ref)[0]
    allData += (specgram.copy().T,)

allData = np.concatenate(allData)

print allData.shape

print "DATA LOAD COMPLETED - Shape = {}".format(allData.shape)

scaler = StandardScaler()

scaler.fit(allData)

if not os.path.exists(DIR_SCALER_OUT):
    os.mkdir(DIR_SCALER_OUT)
joblib.dump(scaler, DIR_SCALER_OUT + SCALER_FILE_NAME)

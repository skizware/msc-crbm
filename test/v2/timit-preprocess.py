import librosa.core as lc
import numpy as np
import scipy
from subprocess import Popen, PIPE
import os

TIMIT_SAMPLE_RATE = 16000

TRAIN_SET_BASE_DIR = '/home/dave/Downloads/timit/TIMIT/TRAIN/'
OUTPUT_DIR = 'TIMIT/pre-processed/{}_{}.npy'


proc = Popen(['find', ('%s' % TRAIN_SET_BASE_DIR), '-iname', '*.wav'], stdout=PIPE)
tmp = proc.stdout.read().split('\n')

maxTimeFrames = 115

max = 0
min = 1000

for file in tmp:
    if file is not '':
        fileIdentifier = file[file.rfind('/D') + 1:file.find('.WAV')]
        print fileIdentifier
        data, sampleRate = lc.load(file, sr=TIMIT_SAMPLE_RATE)
        stftMat = lc.stft(data, n_fft=512, hop_length=128)

        if stftMat.shape[1] > max:
            max = stftMat.shape[1]

        if stftMat.shape[1] < min:
            min = stftMat.shape[1]

        numChunks = stftMat.shape[1] / maxTimeFrames

        for chunk in xrange(0,numChunks):
            chunkArr = stftMat[:, chunk*maxTimeFrames:(chunk+1)*maxTimeFrames]
            print chunkArr.shape
            file_dir = OUTPUT_DIR.format(fileIdentifier, chunk)
            if not os.path.exists(os.path.dirname(file_dir)):
                os.makedirs(os.path.dirname(file_dir))

            mag = np.abs(chunkArr)
            ang = np.unwrap(np.angle(chunkArr))
            result = np.array([mag, ang])
            np.save(file_dir, result)

print "MIN = {} and MAX = {}".format(min, max)

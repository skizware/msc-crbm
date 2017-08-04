import librosa.core as lc
import math
import numpy as np
import scipy
from subprocess import Popen, PIPE
import os

TIMIT_SAMPLE_RATE = 16000

TRAIN_SET_BASE_DIR = '/home/dave/Downloads/timit/TIMIT/TEST/'
OUTPUT_DIR = 'TIMIT_TRAIN_NO_CHUNK/pre-processed/{}.npy'
TEST_OUTPUT_DIR = 'TIMIT_TEST_NO_CHUNK/pre-processed/{}.npy'

CORE_MALE_SET = ['DAB0', 'WBT0', 'TAS1', 'WEW0', 'JMP0', 'LNT0', 'LLL0', 'TLS0', 'BPM0', 'KLT0', 'CMJ0', 'JDH0', 'GRT0',
                 'NJM0', 'JLN0', 'PAM0']

CORE_FEMALE_SET = ['ELC0', 'PAS0', 'PKT0', 'JLM0' ,'NLP0', 'MGD0', 'DHC0', 'MLD0']

proc = Popen(['find', ('%s' % TRAIN_SET_BASE_DIR), '-iname', '*.wav'], stdout=PIPE)
tmp = proc.stdout.read().split('\n')

maxTimeFrames = 92

test_tframe_len = 55

max = 0
min = 1000

for file in tmp:
    if file is not '':
        sub_str = file[:file.rfind('/')]
        sub_str = sub_str[sub_str.rfind('/')+2:]
        print sub_str

        if (str(file).find('SX') is not -1 or str(file).find('SI') is not -1) and (sub_str in CORE_FEMALE_SET or sub_str in CORE_MALE_SET):

            fileIdentifier = file[file.rfind('/D') + 1:file.find('.WAV')]
            print fileIdentifier
            data, sampleRate = lc.load(file, sr=TIMIT_SAMPLE_RATE)
            stftMat = lc.stft(data, n_fft=320, hop_length=160)

            #midPoint = math.floor(stftMat.shape[1] / 2.)
            #start = int(midPoint - math.floor(test_tframe_len / 2.))
            #end = int(midPoint + math.ceil(test_tframe_len / 2.))
            #stftMat = stftMat[:, start:end + 1]

            file_dir = TEST_OUTPUT_DIR.format(fileIdentifier)
            if not os.path.exists(os.path.dirname(file_dir)):
                os.makedirs(os.path.dirname(file_dir))

            mag = np.abs(stftMat)
            ang = np.unwrap(np.angle(stftMat))
            result = np.array([mag, ang])
            np.save(file_dir, result)

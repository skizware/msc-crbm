import librosa.core as lc
import math
import numpy as np
import scipy
from subprocess import Popen, PIPE
import os

TIMIT_SAMPLE_RATE = 16000

TRAIN_SET_BASE_DIR = '/home/dave/Downloads/timit/TIMIT/TEST/'
OUTPUT_DIR = 'TIMIT_TRAIN_NO_CHUNK/pre-processed/{}.npy'
TEST_OUTPUT_DIR = 'TIMIT_TEST_NO_CHUNK_SPEAKER_ID/pre-processed/{}.npy'

CORE_MALE_SET = ['DAB0', 'WBT0', 'TAS1', 'WEW0', 'JMP0', 'LNT0', 'LLL0', 'TLS0', 'BPM0', 'KLT0', 'CMJ0', 'JDH0', 'GRT0',
                 'NJM0', 'JLN0', 'PAM0']

CORE_FEMALE_SET = ['ELC0', 'PAS0', 'PKT0', 'JLM0' ,'NLP0', 'MGD0', 'DHC0', 'MLD0']

proc = Popen(['find', ('%s' % TRAIN_SET_BASE_DIR), '-iname', '*.wav'], stdout=PIPE)
tmp = proc.stdout.read().split('\n')
tmp.sort()

sx_tally_map = {}
total_tally_map = {}

for file in tmp:
    if file is not '':
        sub_str = file[:file.rfind('/')]
        sub_str = sub_str[sub_str.rfind('/')+2:]
        print sub_str

        # if (str(file).find('SX') is not -1 or str(file).find('SI') is not -1) and (sub_str in CORE_FEMALE_SET or sub_str in CORE_MALE_SET):

        speaker_id = file[file.index('/DR') + 5:]
        speaker_id = speaker_id[:speaker_id.index('/')]
        if speaker_id not in total_tally_map.keys():
            total_tally_map[speaker_id] = 0

        fileIdentifier = file[file.rfind('/D') + 1:file.find('.WAV')]
        print fileIdentifier

        if 'SX' in fileIdentifier[fileIdentifier.rindex('/')+1:]:
            if speaker_id not in sx_tally_map.keys():
                sx_tally_map[speaker_id] = 1
                continue
            else:
                sx_tally_map[speaker_id] += 1

            if sx_tally_map[speaker_id] > 3:
                #continue
                pass
            else:
                print "SKIPPING FILE FOR {} DUE TO {}".format(speaker_id, sx_tally_map[speaker_id])
                continue

        else:
            continue

        fileIdentifier = file[file.rfind('/D') + 1:file.find('.WAV')]
        print fileIdentifier
        total_tally_map[speaker_id] += 1
        data, sampleRate = lc.load(file, sr=TIMIT_SAMPLE_RATE)
        stftMat = lc.stft(data, n_fft=320, hop_length=160)

        file_dir = TEST_OUTPUT_DIR.format(fileIdentifier)
        if not os.path.exists(os.path.dirname(file_dir)):
            os.makedirs(os.path.dirname(file_dir))

        mag = np.abs(stftMat)
        ang = np.unwrap(np.angle(stftMat))
        result = np.array([mag, ang])
        np.save(file_dir, result)

for key in total_tally_map.keys():
    print "Speaker {} Total {}".format(key, total_tally_map[key])
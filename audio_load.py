import numpy as np
import pydub
import librosa.core as lc
from matplotlib import pyplot as plt

def load_audio_from_files(file_list, file_format, length, fs):
    naudio = len(file_list)
    nsample = int(fs * (length / 1000))
    audio_samples = np.zeros(shape=(naudio, nsample))

    idx = 0
    for f in file_list:
        print('Loading audio sample: ' + f + '\n')
        audio = pydub.AudioSegment.from_file(f, file_format)

        if audio.frame_rate != fs:
            raise ValueError('Audio sample ' + f + ' has frame rate of: ' + str(audio.frame_rate) + '\n' +
                             'A frame rate of ' + fs + ' was expected\n')
        else:
            long_audio = audio
            while len(long_audio) < length:
                long_audio = long_audio + audio
            trunc_audio = long_audio[:length]
            audio_arr = np.array(trunc_audio.get_array_of_samples(),dtype='float')
            audio_samples[idx] = audio_arr.reshape((1, nsample))
            print('Added audio sample: ' + f + '\n')
            idx = idx + 1

    return audio_samples

def audio2spec(audio_list, window_size, window_overlap, n_fft):
    naudio = len(audio_list)
    nsamples = len(audio_list[0])

    spec_tens = np.zeros(shape=(naudio, 1 + n_fft//2, 1 + nsamples // window_overlap))
    idx = 0

    for audio in audio_list:
        stf = lc.stft(audio, win_length=window_size, hop_length=window_overlap, n_fft=n_fft)
        spec_tens[idx] = lc.stft(audio, win_length=window_size, hop_length=window_overlap, n_fft=n_fft)
        idx = idx + 1

    return spec_tens

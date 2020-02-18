import numpy as np
import pydub
import librosa.core as lc
import librosa.display as ld
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_audio_from_files(file_list, file_format, length, fs):
    """
    Args:
        file_list: list of strings containing paths to audio files
                   All files must have same format.
        file_format: string, one of wav, mp3, flac, or other 
        length: desired length (in ms) of the output
        fs: sampling frequency
    Returns:
        audio_samples: np array of samples, shape=(naudio, nsamples)
    """
    naudio = len(file_list)
    nsample = int(fs * (length / 1000))
    audio_samples = np.zeros(shape=(naudio, nsample))

    for idx,f in enumerate(tqdm(file_list)):
        #print('Loading audio sample: ' + f + '\n')
        audio = pydub.AudioSegment.from_file(f, file_format)

        if audio.frame_rate != fs:
            sound.set_frame_rate(fs)
        else:
            long_audio = audio
            while len(long_audio) < length:
                long_audio = long_audio + audio
            trunc_audio = long_audio[:length]
            audio_arr = np.array(trunc_audio.get_array_of_samples(),dtype='float')
            audio_samples[idx] = audio_arr.reshape((1, nsample))
            # print('Added audio sample: ' + f + '\n')

    return audio_samples

def audio2spec(audio_list, window_size, window_overlap, n_fft):
    """
    Args:
        audio_list: a numpy array of audio samples, with dimensions (naudio, nsample)
        window_size: the size of the stft window, in samples
        window_overlap: amount of window overlap, in samples
        n_fft: size of windowed signal after zero padding
    Returns:
        spec_tens: np array of spectrograms, shape=(naudio, 1 + nfft//2, 1 + nsamples // window_overlap)
    """
    naudio, nsamples = audio_list.shape
    spec_tens = np.zeros(shape=(naudio, 1 + n_fft//2, 1 + nsamples // window_overlap))

    for idx,audio in enumerate(tqdm(audio_list)):
        stf = lc.stft(audio, win_length=window_size, hop_length=window_overlap, n_fft=n_fft)
        spec_tens[idx] = np.abs(lc.stft(audio, win_length=window_size, hop_length=window_overlap, n_fft=n_fft))

    return spec_tens

if __name__ == '__main__':
    file_list = ["14.wav","32.wav"]
    audio_samples = load_audio_from_files(file_list, "wav", 3000, 16000)
    spec_tens = audio2spec(audio_samples, 320, 160, 8000)
    spec1 = spec_tens[1]
    ld.specshow(spec1, x_axis='time', y_axis='hz',sr=16000,hop_length=160)
    plt.colorbar(format='%2.0f dB')
    plt.title('Spectrogram of 32.wav')
    plt.show()

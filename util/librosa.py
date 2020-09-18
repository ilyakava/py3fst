from librosa.filters import get_window
from librosa.util import pad_center

def librosa_window_fn(window_length, n_fft):
    """Provides a window_length hann window padded to be n_fft long
    See: http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/_modules/librosa/core/spectrum.html#stft
    """
    fft_window = get_window('hann', window_length, fftbins=True)
    return pad_center(fft_window, n_fft)
    
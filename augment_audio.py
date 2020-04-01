"""Functions for augmenting audio.

Most code assumes 16kHz sampling rate.
"""

from collections import namedtuple

import soundfile as sf
from pyrubberband import pyrb
import librosa.effects
import librosa.core
import librosa.filters
import numpy as np
import pydub
import pyroomacoustics as pra
import scipy.fftpack

from util.ft import normalize_0_1, dct_filters

import pdb

sr=16000

# struct for audio content. Main idea is that content is between start and end
# but leading/trailing audio is present in samples.
# Clip = namedtuple('Clip', ['samples', 'start', 'end'])

class Clip(object):
    
    def __init__(self, samples, start, end):
        """
        start and end should mark the "content" that should be included.
        For wakewords this is the audible start and end of the word.
        For padding audio, this can encapsulate silence, as that is
        useful "content" for the purpose of padding.
        """
        self.samples = samples
        self.start = start
        self.end = end
        assert start >= 0 and end <= len(samples), 'Content must be within clip'

def filter_for_speech(samples, sr=sr):
    """Filters np.array of samples with speech bandpass filter.
    """
    samples = (samples * 2**15).astype(np.int16)
    sound = pydub.AudioSegment(
        samples.tobytes(), 
        frame_rate=sr,
        sample_width=samples.dtype.itemsize, 
        channels=1
    )
    
    sound_hi = pydub.effects.high_pass_filter(sound, 85)
    sound_mid = pydub.effects.low_pass_filter(sound_hi, 255)
    y_ = sound_mid.get_array_of_samples()
    y = np.array(y_, dtype=np.float) / np.abs(y_).max()
    return y
    
def pad_clip(clip, padding=(0,0)):
    pad_left, _ = padding
    y = np.pad(clip.samples, pad_width=padding, mode='constant', constant_values=0)
    return Clip(y, clip.start+pad_left, clip.end+pad_left)

def extract_other_speech_padding(samples_bg, length=80000):
    """
    Returns:
      leading Clip and trailing Clip
    """
    # try to figure out start time with filtered speech
    samples_filtered = filter_for_speech(samples_bg)
    audio_chunks = librosa.effects.split(samples_filtered, top_db=20, frame_length=320, hop_length=160)
    audio_chunks_flat = [item for sublist in audio_chunks for item in sublist]
    start = min(audio_chunks_flat)
    end = max(audio_chunks_flat)
    # print(start,len(samples_bg)-end)
    # TODO: check that start/end times are "correct," maybe close enough to front/end
    middle = (end-start) // 2
    
    if end - middle > length:
        lead = Clip(samples_bg, end - length, end)
    else: # in this case the clip is too short
        lead = Clip(samples_bg, middle, end)
        lead = pad_clip(lead, (length-(end - middle),0))
        lead.start = 0
    
    if middle - start > length:
        trail = Clip(samples_bg, start, start + length)
    else: # in this case the clip is too short
        trail = Clip(samples_bg, start, middle)
        trail = pad_clip(trail, (0, length - (middle - start)))
        trail.end = len(trail.samples)
        
    return lead, trail
    
def align_three(source_a, source_b, source_c, b_delay=0, c_delay=0):
    """
    Makes a multitrack np.array of the samples in 3 sources.
    
    
    Args:
      sources should all be of Clip type
      source_a is cut at its start, as much trailing content is included as fits
      source_c is cut at its end, as much leading content is included as fits
      source_b is not cut at all, as much leading and trailing content is included as fits
      b_delay: number samples to delay b
      c_delay: number samples to delay c
    Returns:
      np.array of output, channels=axis 1. np.array of content start end times, shape 2x3.

    There is the concept of content, track which includes padding for content, and output.

    Turn the source clips into tracks, stick the tracks in the output mux.
    """
    a_len = source_a.end - source_a.start
    b_len = source_b.end - source_b.start
    c_len = source_c.end - source_c.start
    a_start_in_out = 0
    b_start_in_out = a_start_in_out + a_len + b_delay
    c_start_in_out = b_start_in_out + b_len + c_delay
    out_len = c_start_in_out + c_len

    output = np.zeros((out_len, 3))
    labels = np.zeros((2, 3), dtype=int)

    # start cutting and fitting
    a_pad_right_len = len(source_a.samples) - source_a.end
    if a_start_in_out + a_len + a_pad_right_len > out_len:
        a_pad_right_len = out_len - (a_start_in_out + a_len)
    a_track = source_a.samples[source_a.start:(source_a.end + a_pad_right_len)]
    output[a_start_in_out:(a_start_in_out + a_len + a_pad_right_len),0] = a_track
    labels[:,0] = [a_start_in_out,a_start_in_out + a_len]

    b_pad_left_len = source_b.start
    if b_start_in_out - b_pad_left_len < 0:
        b_pad_left_len = b_start_in_out
    b_pad_right_len = len(source_b.samples) - source_b.end
    if b_start_in_out + b_len + b_pad_right_len > out_len:
        b_pad_right_len = out_len - (b_start_in_out + b_len)
    b_track = source_b.samples[(source_b.start - b_pad_left_len):(source_b.end + b_pad_right_len)]
    output[(b_start_in_out - b_pad_left_len):(b_start_in_out + b_len + b_pad_right_len),1] = b_track
    labels[:,1] = [b_start_in_out,b_start_in_out + b_len]

    c_pad_left_len = source_c.start
    if c_start_in_out - c_pad_left_len < 0:
        c_pad_left_len = c_start_in_out
    c_track = source_c.samples[(source_c.start-c_pad_left_len):source_c.end]
    output[(c_start_in_out-c_pad_left_len):(c_start_in_out + c_len), 2] = c_track
    labels[:,2] = [c_start_in_out,c_start_in_out + c_len]
    
    return output, labels
    
def align_two(source_a, source_b, b_delay=0):
    """
    Makes a multitrack np.array of the samples in 2 sources.
    
    
    Args:
      sources should all be of Clip type
      source_a is cut at its start, as much trailing content is included as fits
      source_c is cut at its end, as much leading content is included as fits
      b_delay: number samples to delay b
    Returns:
      np.array of output, channels=axis 1. np.array of content start end times, shape 2x2.

    There is the concept of content, track which includes padding for content, and output.

    Turn the source clips into tracks, stick the tracks in the output mux.
    """
    a_len = source_a.end - source_a.start
    b_len = source_b.end - source_b.start
    a_start_in_out = 0
    b_start_in_out = a_start_in_out + a_len + b_delay
    out_len = b_start_in_out + b_len

    output = np.zeros((out_len, 2))
    labels = np.zeros((2, 2), dtype=int)

    # start cutting and fitting
    a_pad_right_len = len(source_a.samples) - source_a.end
    if a_start_in_out + a_len + a_pad_right_len > out_len:
        a_pad_right_len = out_len - (a_start_in_out + a_len)
    a_track = source_a.samples[source_a.start:(source_a.end + a_pad_right_len)]
    output[a_start_in_out:(a_start_in_out + a_len + a_pad_right_len),0] = a_track
    labels[:,0] = [a_start_in_out,a_start_in_out + a_len]

    b_pad_left_len = source_b.start
    if b_start_in_out - b_pad_left_len < 0:
        b_pad_left_len = b_start_in_out
    b_track = source_b.samples[(source_b.start-b_pad_left_len):source_b.end]
    output[(b_start_in_out-b_pad_left_len):(b_start_in_out + b_len), 1] = b_track
    labels[:,1] = [b_start_in_out,b_start_in_out + b_len]
    
    return output, labels
    

    
def extract_example(samples, center_chunk_start, example_length, n_right_chunks=10, n_left_chunks=20):
    """
    This is for extracting audio that will be a certain shape
    when a spectrogram is made. Each chunk will correspond to 1 or
    more columns in the spectrogram. Therefore it's best to think
    of chunk_length in terms of multiples of hop size.
    10ms = 160 samples.
    
    Args:
      center_chunk_start: idx of center in samples
      example_length: in samples, should be exactly chunkable into the number
        of chunks specified.
    Returns:
      samples cut to size
    """
    n_tot_chunks = 1+n_left_chunks+n_right_chunks
    assert example_length % n_tot_chunks == 0, 'Length (%i) must be exactly divisble by number of chunks (%i).' % (example_length, n_tot_chunks)
    chunk_length = example_length // (n_tot_chunks)
    slice_right = center_chunk_start + (1+n_right_chunks)*chunk_length
    slice_left = center_chunk_start - (n_left_chunks*chunk_length)
    assert slice_left > 0, 'Not enough left padding in input'
    assert slice_right < len(samples), 'Not enough right padding in input'
    return samples[slice_left:slice_right]

def get_chunk(samples, chunk):
    return np.array(samples[chunk[0]:chunk[1]])
    
def time_stretch_to_target(clip, target_len=0.550, sr=sr, tolerance=0.05):
    """
    Don't alter if current length is within tolerance.
    Args:
      clip: of Clip type
      target_len: length in miliseconds that clip content should be
      tolerance: in ms
    """
    cur_len = (clip.end - clip.start)/sr
    if abs(cur_len - target_len) < tolerance:
        return clip
    else:
        time_stretch = cur_len / target_len
        new_samples = pyrb.time_stretch(clip.samples, sr=sr, rate=time_stretch)
        new_start = int(clip.start / time_stretch)
        new_end = int(clip.end / time_stretch)
        return Clip(new_samples, new_start, new_end)


def augment_audio(p_file, p_start_end, n_file, p_duration, pitch_shift, silence_1, silence_2, loudness):
    """
    Does everything but extraction. Outputs multiple channels.
    
    Args:
      p_duration: in ms
      overlap_1: silence in ms between speech before wakeword and wakeword, if negative there is overlap
      overlap_2: silence in ms between wakeword and speech after wakeword, if negative there is overlap
      pitch_shift: octaves to shift, can be float, -3..+3 is a reasonable range
    """
    
    wakeword_samples, _ = sf.read(p_file)
    wakeword = Clip(wakeword_samples, *p_start_end)
    wakeword = time_stretch_to_target(wakeword, target_len=p_duration, sr=sr, tolerance=0.05)
    wakeword = Clip(pyrb.pitch_shift(wakeword.samples, sr=sr, n_steps=pitch_shift), wakeword.start, wakeword.end)
    
    samples_bg, _ = sf.read(n_file)
    lead, trail = extract_other_speech_padding(samples_bg)
    
    output, labs = align_three(lead, wakeword, trail, int(silence_1*sr), int(silence_2*sr))
    
    def get_power(clip):
        return np.sqrt((clip.samples[clip.start:clip.end]**2).sum() / (clip.end-clip.start))
    k = (get_power(lead)+get_power(trail)) / (2*get_power(wakeword))
    
    output[:,1] *= k*loudness
    return output, labs
    
def get_chunk(samples, chunk):
    return np.array(samples[chunk[0]:chunk[1]])

def augment_audio_two_negatives(n_file1, n_file2, silence, loudness):
    """
    Args:
      silence: in ms
    """
    a, _ = sf.read(n_file1)
    b, _ = sf.read(n_file2)
    
    _, trail = extract_other_speech_padding(a)
    lead, _ = extract_other_speech_padding(b)
    output, labs = align_two(lead, trail, int(silence*sr))
    
    def get_power(clip):
        return np.sqrt((clip.samples[clip.start:clip.end]**2).sum() / (clip.end-clip.start))
    k = get_power(lead) / get_power(trail)
    output[:,1] *= k*loudness

    return output, labs

def gwn_for_audio(y, snr=65.0):
    """Gaussian White Noise.
    """
    gwn_power = np.sqrt((y**2).mean()) / snr
    gwn = np.random.normal(0,gwn_power,len(y))
    return gwn

# Audio mixing with room sim constants
standing_height = 1.65
siting_height = 1.2
speaker_height = standing_height # no big difference
table_height = 0.75
dist_to_wall = 0.6 # 0.3 is one foot

def mix_2_sources(source1, source2, room_dim, room_absorption, source1_to_mic_rel_distance):
    """Use a room sim to mix two sources of audio.
    
    Room Arrangement:
    The room is a cube. and  source1-Mic-source2 is set to be
    an equilateral triangle at first. Then source1 is walked
    towards the mic according to source1_to_mic_rel_distance.
    1.0 means it is not walked to the mic at all. 0.05 means it
    is on top of the mic. The room mixer adjusts the volume
    automatically. With a value of 0.01 source2 is barely audible. 
    
    Args:
        room_dim: [x, y, z] dims in meters of room
        room_absorption: float 0 to 1, 0 is high reflection, extreme
            values are noticible but otherwise is subtle.
        source1_to_mic_rel_distance: float 0 to 1, described above.
        source1: mono audio samples assumed to be already aligned
            with source2 which is assumed to be the same shape.
    Returns:
        mono mixed sound
    """
    corners = np.array([[0,0], [0,room_dim[0]], [room_dim[1],room_dim[0]], [room_dim[1],0]]).T
    height = room_dim[2]
    pos1 = np.array([dist_to_wall, dist_to_wall, speaker_height])
    pos2 = np.array([room_dim[1]-dist_to_wall, dist_to_wall, speaker_height])
    mic_radius = .005
    # when mic array is this small, no big difference since
    # we ignore direction here
    n_mic = 1
    mic_pos = np.array([room_dim[1]/2, room_dim[0], table_height])
    
    pos1 = (1-source1_to_mic_rel_distance)*mic_pos + source1_to_mic_rel_distance*pos1
    
    room = pra.Room.from_corners(corners, fs=16000, absorption=room_absorption, max_order=5)
    room.extrude(height)
    room.add_source(pos1, signal=source1)
    room.add_source(pos2,signal=source2)
    R = pra.circular_2D_array(mic_pos[:2], n_mic, 0, mic_radius)
    R = np.concatenate((R, np.ones((1, n_mic)) * mic_pos[2]), axis=0)
    mics = pra.MicrophoneArray(R, sr)
    room.add_microphone_array(mics)
    
    room.simulate()
    mono_room = np.mean(room.mic_array.signals, axis=0)
    return mono_room

def samples2spectrogam(samples, win_length, hop_length, n_fft=512):
    spec = np.abs(librosa.core.stft(samples,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=n_fft))
    height = n_fft // 2
    width = len(samples) // hop_length
    spec = spec[-height:,-width:]
    
    spec_db = librosa.amplitude_to_db(spec)
    spec_db = np.clip(spec_db, -55, 65)
    spec = librosa.db_to_amplitude(spec_db)
    spec = normalize_0_1(spec)
    
    return spec

def samples2mfcc(samples, win_length, hop_length, n_fft=512, n_mels=80, n_mfcc=40):
    """Like in deep-voice-conversion
    (Which only uses mfcc)
    """
    
    samples = librosa.effects.preemphasis(samples, coef=0.97)
    
    mag = np.abs(librosa.core.stft(samples,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=n_fft))
        
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels) 
    mel = np.dot(mel_basis, mag)
    
    mel_db = librosa.amplitude_to_db(mel)
    mfccs = np.dot(dct_filters(n_mfcc, n_mels), mel_db)
    
    height = n_mfcc
    width = len(samples) // hop_length
    
    out = mfccs[-height:,-width:]
    
    return out
    
    # if we were to output mel instead:
    # normalize mel only
    # mel_db = normalize_0_1(mel_db, -55, 35)
    # out = mel_db[-height:,-width:] # border effect is visible in first 2 columns in width
    # return out
    
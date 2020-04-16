#!/usr/bin/env python3
"""
Stream inference on microphone data.

Text-mode spectrogram using live microphone data from:
https://python-sounddevice.readthedocs.io/en/0.3.15/examples.html#real-time-text-mode-spectrogram

Usage:
python stream_inference.py --model_dir /Users/ilyak/Downloads/spectrogram03.20.20/1584800014
"""

import argparse
import math
import shutil
import sys
from time import sleep, monotonic

import cv2
from librosa import amplitude_to_db, db_to_amplitude
from librosa.filters import get_window
from librosa.util import pad_center
import numpy as np
import sounddevice as sd
import tensorflow as tf

import pdb

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

class MovingAvgPerf():
  def __init__(self, nticks=10):
    self.times = []
    self.nticks = nticks

  def tick(self, diff):
    self.times.append(diff)
    if len(self.times) > self.nticks:
      self.times.pop(0)

  def fps_str(self):
    fps = len(self.times) / sum(self.times)
    return '%.2f fps' % fps

class MovingWindowPerf(MovingAvgPerf):

  def tick(self):
    super().tick(monotonic())

  def fps_str(self, text='fps'):
    if len(self.times) == 1:
      fps = 0
    else:
      fps = len(self.times) / (self.times[-1] - self.times[0])
    return '%.2f %s' % (fps, text)

def __draw_label(img, text, bg_color, pos=(0,0)):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    pos = (pos[0] + 5, pos[1] + int(txt_size[0][1]*2))

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

# ANSI text version
# Stolen from https://gist.github.com/maurisvh/df919538bcef391bc89f
# colors = 30, 34, 35, 91, 93, 97
# chars = ' :%#\t#%:'
# gradient = []
# for bg, fg in zip(colors, colors[1:]):
#     for char in chars:
#         if char == '\t':
#             bg, fg = fg, bg
#         else:
#             gradient.append('\x1b[{};{}m{}'.format(fg, bg + 10, char))

# samplerate = sd.query_devices(args.device, 'input')['default_samplerate']
buffer_size_s = 10
samplerate = 16000
window_length = int(0.025 * samplerate)
n_fft = 512
hop_length = int(0.01 * samplerate)

samples_buffer_block_size = hop_length
samples_buffer_nblocks = 1 + n_fft // samples_buffer_block_size
samples_buffer = np.zeros(samples_buffer_nblocks * samples_buffer_block_size)
samples_buffer_p = 0

spec_buffer_w = buffer_size_s * samplerate // samples_buffer_block_size
# Also add some historical padding
# write in 2 locations if within the last N seconds
# of buffer
buffer_pad_size_s = 2
spec_buffer_h = 256
spec_buffer_pad = buffer_pad_size_s * samplerate // samples_buffer_block_size
spec_buffer = np.zeros((spec_buffer_h, spec_buffer_pad + spec_buffer_w))
spec_buffer_p = 0

# modeled after: https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#stft
fft_window = get_window('hann', window_length, fftbins=True)
fft_window = pad_center(fft_window, n_fft)

# normalize audio.
# should match how network was trained.
norm_win = get_window('hann', 5, fftbins=False)
norm_win = norm_win / norm_win.sum()
target_dBFS = -15.0

columns = 80


def update_spectrogram(indata, frames, time, status):
    global samples_buffer, spec_buffer, samples_buffer_p, spec_buffer_p

    if status:
        text = ' ' + str(status) + ' '
        print('\x1b[34;40m', text.center(columns, '#'),
              '\x1b[0m', sep='')
    if any(indata):
        
        indata = np.array(indata[:,0])

        # cur_dBFS = np.convolve(indata, norm_win, 'same').max()
        # a = 10**( (target_dBFS-cur_dBFS) / 20.0 )
        normed_indata = indata
        if samples_buffer_p < (samples_buffer_nblocks - 1):
            print('buffering' + ('.'*samples_buffer_p))
            ss = samples_buffer_p * samples_buffer_block_size
            se = (samples_buffer_p + 1) * samples_buffer_block_size
            samples_buffer[ss:se] = normed_indata
            samples_buffer_p += 1
        elif samples_buffer_p == (samples_buffer_nblocks - 1):
            ss = samples_buffer_p * samples_buffer_block_size
            se = (samples_buffer_p + 1) * samples_buffer_block_size
            samples_buffer[ss:se] = normed_indata
            # fft
            magnitude = np.abs(np.fft.rfft(fft_window * samples_buffer[-n_fft:], n=n_fft))
            #
            mag_db = amplitude_to_db(magnitude)
            mag_db = np.clip(mag_db, -55, 65)
            magnitude = db_to_amplitude(mag_db)
            
            # primary write
            write_idx = (spec_buffer_p % spec_buffer_w)
            spec_buffer[:, spec_buffer_pad + write_idx] = magnitude[:n_fft//2]
            # secondary buffer write
            if spec_buffer_w < write_idx + spec_buffer_pad:
                pad_write_idx = (write_idx + spec_buffer_pad) % spec_buffer_w
                spec_buffer[:, pad_write_idx] = magnitude[:n_fft//2]

            spec_buffer_p += 1
            samples_buffer = np.roll(samples_buffer, -samples_buffer_block_size)

            # ANSI text version
            # ds = len(magnitude) // columns
            # line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))]
            #         for x in magnitude[::ds])
            # print(*line, sep='', end='\x1b[0m\n')
        else:
            raise ValueError('samples_buffer_p out of range, is %i' % samples_buffer_p)
    else:
        print('no input')

def stream_spectrogram_of_microphone_audio(args):
    with sd.InputStream(device=args.device, channels=1, callback=update_spectrogram,
                        blocksize=samples_buffer_block_size,
                        samplerate=samplerate):

        while True:
            sleep(0.02)

            cv2.imshow("Press 'q' to quit", np.asarray((spec_buffer * 255).astype(np.uint8)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def stream_inference_of_microphone_audio(args):
    """
    The spectrum sits in a buffer with width spec_buffer_pad + spec_buffer_w .
    The first spec_buffer_pad of it is a copy of the last spec_buffer_pad of it.
    """
    with sd.InputStream(device=args.device, channels=1, callback=update_spectrogram,
                        blocksize=samples_buffer_block_size,
                        samplerate=samplerate):
        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.model_dir)
            predictor = tf.contrib.predictor.from_saved_model(args.model_dir)

            network_spec_w = args.model_input_width
            spectrogram_predictions = np.zeros((spec_buffer_w + spec_buffer_pad, 3))
            
            display_predictions = np.stack([np.arange(spec_buffer_w), np.zeros(spec_buffer_w)]).astype(int).T
            frame = np.zeros((spec_buffer_h, spec_buffer_w, 3), dtype=np.uint8)

            alpha = 0.025
            N = 90
            myfilt = alpha*((1-alpha)**np.arange(0,N))
            myfilt /= myfilt[:60].sum()

            last_pred_write = 0

            perf = MovingWindowPerf()
            while True:
                # sleep(0.01) # restrict max fps to 100
                imageify = spec_buffer[:,spec_buffer_pad:].copy()
                imageify = np.log(1+imageify)
                imageify = (imageify - imageify.min()) / (imageify.max() - imageify.min())
                imageify = (imageify * 255).astype(np.uint8)
                frame[:,:,0] = imageify
                frame[:,:,1] = imageify
                frame[:,:,2] = imageify

                idx_now = spec_buffer_p % spec_buffer_w
                # we look into the past
                se = idx_now + spec_buffer_pad
                ss = se - network_spec_w

                next_input = np.expand_dims(spec_buffer[:, ss:se], 0)

                prediction = predictor({"spectrograms": next_input })['output']
                perf.tick()
                prediction = prediction[0] # batch size of one
                
                spectrogram_predictions[last_pred_write:se,:] = prediction[-1,:] # write latest prediction
                last_pred_write = se
                pred_class = np.argmax(prediction[-1,:])
                
                # secondary buffer write because of the padding
                # if ss < spec_buffer_pad:
                #     spectrogram_predictions[ss+spec_buffer_w:spec_buffer_pad+spec_buffer_w] = spectrogram_predictions[ss:spec_buffer_pad]
                
                # erase the future
                spectrogram_predictions[se+1:] = 0

                # smoothed_mask = np.correlate(spectrogram_predictions, myfilt, 'same')

                ### display code
                # display_predictions[:,1] = (spec_buffer_h - (smoothed_mask[spec_buffer_pad:] * spec_buffer_h)).astype(int)
                # cv2.polylines(frame, [display_predictions], isClosed=False, color=(0,0,255))

                blue = (255,0,0)
                red = (0,0,255)
                green = (0,255,0)
                colors = [green, blue, red]
                activities = ['voice', 'silence', 'alexa']

                for i, color in enumerate(colors):
                    display_predictions[:,1] = (spec_buffer_h - (spectrogram_predictions[spec_buffer_pad:, i] * spec_buffer_h)).astype(int)
                    cv2.polylines(frame, [display_predictions], isClosed=False, color=color) 

                cv2.line(frame, (idx_now, 0), (idx_now, spec_buffer_h), green, 2) # moving vertical line
                cv2.line(frame, (0, spec_buffer_h//2), (spec_buffer_w, spec_buffer_h//2), red, 2) # horizontal line

                __draw_label(frame, activities[pred_class], colors[pred_class], (spec_buffer_w//2, 0))
                __draw_label(frame, perf.fps_str('inferences/sec'), green)

                cv2.imshow("Press 'q' to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

def main():
    usage_line = ' press q to quit '
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__ + '\n\nSupported keys:' + usage_line,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-m', '--model_dir', type=str, default='/Users/ilyak/Downloads/spectrogram03.20.20/1584800014',
        help='...')
    parser.add_argument(
        '-w', '--model_input_width', type=int, default=31,
        help='This is the width of the spectrogram input the model expects. Used to be 124.')

    args = parser.parse_args(remaining)

    print(usage_line)
    stream_inference_of_microphone_audio(args)

if __name__ == '__main__':
    main()

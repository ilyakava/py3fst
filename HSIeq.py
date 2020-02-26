"""Creates a false color image from HSI data by averaging neighboring bands.
Launches a GUI where gaussian weighted neigboring bands in a user
defined window are averaged together to form each new RGB band.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import datetime

import os
import scipy.io as sio

import pdb

DATASET_PATH = '/scratch0/ilya/locDoc/data/hyperspec/datasets'

# dataset_name, data_struct_field_name = ['Smith_117chan.mat', 'Smith']
ds = 2 # downsample factor

# dataset_name, data_struct_field_name = ['Indian_pines_corrected.mat', 'indian_pines_corrected']

dataset_name, data_struct_field_name = ['Salinas_corrected.mat', 'salinas_corrected']

# dataset_name, data_struct_field_name = ['KSC_corrected.mat', 'KSC']

# dataset_name, data_struct_field_name = ['Botswana.mat', 'Botswana']

# dataset_name, data_struct_field_name = ['PaviaU.mat', 'paviaU']

# dataset_name, data_struct_field_name = ['Pavia_center_right.mat', 'Pavia_center_right']


def normalize_channels(cube):
    chan_maxes = np.expand_dims(np.expand_dims(np.max(np.max(cube,0),0),0),0)
    cube /= chan_maxes
    return cube

mat_contents = sio.loadmat(os.path.join(DATASET_PATH, dataset_name))
data = mat_contents[data_struct_field_name].astype(np.float32)
orig_data = data
dataset_wvlenghts = {
    'PaviaU.mat': [430, 860, 103],
    'Pavia_center_right.mat': [430, 860, 102],
    'Indian_pines_corrected.mat': [400, 2500, 200],
    'Salinas_corrected.mat': [400, 2500, 204],
    'Smith_117chan.mat': [445, 2486, 117],
    'Botswana.mat': [400, 2500, 145]
}

# downsample cube
data = data[::ds, ::ds, :]


minwave, maxwave, nbands = dataset_wvlenghts[dataset_name]

rangewave = maxwave - minwave
stepwave = rangewave / float(nbands)
maxamp = 1

fig, axes = plt.subplots(1, 2)
ax = axes[0]
axim = axes[1]
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.linspace(minwave, maxwave, nbands)
a0 = 1
v0 = 200
m0 = 1000

# initial values from:
# https://en.wikipedia.org/wiki/Color#/media/File:Cones_SMJ2_E.svg
state = {
    'red': {
        'm1': 580,
        'v1': 60,
        'a1': 1,
    },
    'blue': {
        'm1': 440,
        'v1': 25,
        'a1': 1,
    },
    'green': {
        'm1': 540,
        'v1': 45,
        'a1': 1,
    },
    'color': 'red'
}

def params_to_line(state_at_color):
    mean = state_at_color['m1']
    var = state_at_color['v1']
    amp = state_at_color['a1']

    return amp * np.exp(-(t - mean)**2 / (2*var**2))

sred = params_to_line(state['red'])
sblue = params_to_line(state['blue'])
sgreen = params_to_line(state['green'])

l, = ax.plot(t, sred, lw=2, color='red')
lblue, = ax.plot(t, sblue, lw=2, color='blue')
lgreen, = ax.plot(t, sgreen, lw=2, color='green')

avg_spectrum = data.sum(axis=(0,1))
avg_spectrum /= avg_spectrum.max()
ax.plot(t, avg_spectrum, lw=1, color='black')

lines = {
    'red': l,
    'blue': lblue,
    'green': lgreen,
}

ax.axis([minwave, maxwave, 0, maxamp])

axcolor = 'lightgoldenrodyellow'
axmean = plt.axes([0.25, 0.1, 0.65, 0.018], facecolor=axcolor)
axvar = plt.axes([0.25, 0.15, 0.65, 0.018], facecolor=axcolor)
axamp = plt.axes([0.25, 0.05, 0.65, 0.018], facecolor=axcolor)
resetax = plt.axes([0.8, 0, 0.1, 0.025])
saveax = plt.axes([0.6, 0, 0.1, 0.025])

smean = Slider(axmean, 'Mean', minwave, maxwave, valinit=m0)
svar = Slider(axvar, 'Var', stepwave, rangewave, valinit=v0)
samp = Slider(axamp, 'Amp', 0.1, maxamp, valinit=a0)


def make_img(state):
    red_weights = params_to_line(state['red'])
    blue_weights = params_to_line(state['blue'])
    green_weights = params_to_line(state['green'])

    blue = np.expand_dims(np.sum(blue_weights*data,axis=2),-1)
    green = np.expand_dims(np.sum(green_weights *data,axis=2),-1)
    red = np.expand_dims(np.sum(red_weights *data,axis=2),-1)

    color = normalize_channels(np.concatenate([red, green,blue], axis=2))
    color[:,:,0] *= state['red']['a1']
    color[:,:,1] *= state['blue']['a1']
    color[:,:,2] *= state['green']['a1']
    return color

im = axim.imshow(make_img(state))

def update(val):
    var = svar.val
    mean = smean.val
    amp = samp.val

    state_color = state['color']

    state[state_color]['m1'] = mean
    state[state_color]['v1'] = var
    state[state_color]['a1'] = amp

    s = params_to_line(state[state_color])
    
    lines[state_color].set_ydata(s)

    im.set_data(make_img(state))

    fig.canvas.draw_idle()
smean.on_changed(update)
svar.on_changed(update)
samp.on_changed(update)


button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
savebutton = Button(saveax, 'Save', color=axcolor, hovercolor='0.975')


def reset(event):
    smean.reset()
    svar.reset()
button.on_clicked(reset)

def save(event):
    red_weights = params_to_line(state['red'])
    blue_weights = params_to_line(state['blue'])
    green_weights = params_to_line(state['green'])

    blue = np.expand_dims(np.sum(blue_weights*orig_data,axis=2),-1)
    green = np.expand_dims(np.sum(green_weights *orig_data,axis=2),-1)
    red = np.expand_dims(np.sum(red_weights *orig_data,axis=2),-1)

    color = normalize_channels(np.concatenate([red, green,blue], axis=2))
    color[:,:,0] *= state['red']['a1']
    color[:,:,1] *= state['blue']['a1']
    color[:,:,2] *= state['green']['a1']

    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    plt.imsave('/scratch0/ilya/locDownloads/%s_color_%s.png' % (data_struct_field_name, timestamp), color)
    print('saved')
savebutton.on_clicked(save)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    state['color'] = label

    mean = state[label]['m1']
    var = state[label]['v1']
    amp = state[label]['a1']

    s = params_to_line(state[label])
    lines[label].set_ydata(s)

    svar.set_val(var)
    smean.set_val(mean)
    samp.set_val(amp)

    fig.canvas.draw_idle()
colorfunc(state['color']) # run for init
radio.on_clicked(colorfunc)

plt.show()
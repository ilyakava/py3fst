from datetime import datetime
from time import sleep
import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # Sample rate

text = input("How many seconds would you like to record audio for?:")
seconds = int(text)
if not (seconds >= 1 and seconds < 60):
    raise ValueError("Please enter a number between 1 and 60.")

for i in reversed(range(1,4)):
    print('Starting Audio Recording in %i seconds.' % i)
    sleep(1)

print('Recording now...')
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
print('Done Recording Audio.')

timestamp = str(datetime.now().strftime("%m-%d-%Hh%Mm%S"))

outfile = 'voice_recording_%s.wav' % timestamp
write(outfile, fs, myrecording)
print('Wrote %s' % 'voice_recording_%s.wav' % timestamp)

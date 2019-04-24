"""
This example is provided to test the installed package.
The package should be installed from PyPi using pip install speechpy.
"""

import scipy
import scipy.io.wavfile as wav
from scipy.io.wavfile import read
import numpy as np
import speechpy
import os
import soundfile as sf
import pyloudnorm as pyln
import matplotlib
import matplotlib.pyplot as plt
import pitch
import pandas as pd
import pickle
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import winsound

nombre= 'extrastole__151_1306779785624_B.wav'
np.set_printoptions(threshold=np.inf)
# Reading the sample wave file
file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         nombre)
fs, signal = wav.read(file_name)
signal = signal[:,]
#Procesamiento de los datos del audio
# Pre-emphasizing.
signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)

# Staching frames
frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs,
                                          frame_length=0.020,
                                          frame_stride=0.01,
                                          filter=lambda x: np.ones((x,)),
                                          zero_padding=True)

############# Extract MFCC features #############
mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs,
                             frame_length=0.020, frame_stride=0.01,
                             num_filters=40, fft_length=512, low_frequency=0,
                             high_frequency=None)
print('mfcc =', np.mean(mfcc,axis=0))
mfccf=np.mean(mfcc,axis=0)
print('tamano=', len(mfcc[0]))

############# Extract logenergy features #############
logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs,
                                  frame_length=0.020, frame_stride=0.01,
                                  num_filters=13, fft_length=512,
                                  low_frequency=0, high_frequency=None)
print('logenergy', np.mean(logenergy))
loge=np.mean(logenergy)

#Extracción de Caracteristica Pitch
data, rate = sf.read(file_name) # load audio
meter = pyln.Meter(rate) # create BS.1770 meter
loudness = meter.integrated_loudness(data) # measure loudness
print('loudness=', loudness)
p=pitch.find_pitch(file_name)
print('pitch =', p)
#Generando Vector de Datos del Audio
dato=np.append(loge,mfccf)
dato=np.append(dato,loudness)
dato=np.append(dato,p)
print('dato', dato)

#Carga de Clasificador
pkl_filename ="pickle_model.pkl"

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

print(pickle_model.predict([dato]))


#Graficación del Audio
# read audio samples
input_data = read(file_name)
audio = input_data[1]
# plot the first 1024 samples
plt.plot(audio[0:3000])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title
plt.title("Sample Wav")
# display the plot
#Reproducción del Audio
winsound.PlaySound(nombre, winsound.SND_FILENAME)
plt.show()












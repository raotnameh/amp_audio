import numpy as np
from scipy.interpolate import interp1d
from scipy.io import wavfile
import librosa.display
import matplotlib.pyplot as plt

def apply_transfer(signal, transfer, interpolation='linear'):
    constant = np.linspace(-1, 1, len(transfer))
    interpolator = interp1d(constant, transfer, interpolation)
    return interpolator(signal)


# hard limiting
def limiter(x, treshold=0.8):
    transfer_len = 1000
    transfer = np.concatenate([ np.repeat(-1, int(((1-treshold)/2)*transfer_len)),
                                np.linspace(-1, 1, int(treshold*transfer_len)),
                                np.repeat(1, int(((1-treshold)/2)*transfer_len)) ])
    return apply_transfer(x, transfer)

# smooth compression: if factor is small, its near linear, the bigger it is the
# stronger the compression
def arctan_compressor(x, factor=2):
    constant = np.linspace(-1, 1, 1000)
    transfer = np.arctan(factor * constant)
    transfer /= np.abs(transfer).max()
    return apply_transfer(x, transfer)

def plot_out(updated_audio,sr):
    plt.Figure()
    librosa.display.waveplot(updated_audio,sr)
def read(path):
    sr, audio = wavfile.read(path)
    return audio*1.0, sr
# INPUT AUDIO FORMAT SHOULD BE WAV
# path = input path of the audio
# amp_factor = amplifyign factor to audio
# plot = to plot the output audio or no
# save = to save the output audio or no
def run(path,amp_factor = 2, plot = False, save = None):
    audio, sr = read(path)
    audio = audio / np.abs(audio).max() # x scale between -1 and 1

    updated_audio = limiter(audio)
    if save != None:
        wavfile.write(save+ ".wav", sr, updated_audio*amp_factor)
    
    if plot == True:
        plot_out(updated_audio,sr)
    return updated_audio, sr
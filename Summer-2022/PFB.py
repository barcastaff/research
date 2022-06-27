import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 5]


def PFB(tap, signal, fs):
    '''
    Arguments:
        tap: the number of taps of the PFB;
        signal: the signal that needs to be processed
        fs: the sampling rate of the signal (samples per second)

    Returns:
        FFT_sum: the real-DFT of the signal
        FFT_freq: the absolute frequency (in Hz), on the x-axis
    '''
    N = len(signal)
    if N % tap != 0:  # cut the signal short if modulo is not zero
        signal = signal[:N - (N % tap)]
        N = N - (N % tap)
    t = np.arange(N)  # x-axis in terms of frequency channels
    t_sinc = np.linspace(-1, 1, N)  # generation for the sinc function
    sinc = np.sinc(tap / 2 * t_sinc)

    conv = sinc * signal  # convolution step
    conv_list = []
    for i in range(tap):
        conv_list.append(np.array(conv[i * (N // tap):(i + 1) * (N // tap)]))
    conv_array = np.array(conv_list)
    conv_sum = np.sum(conv_array, axis=0)

    FFT_sum = np.fft.rfft(conv_sum)  # real-FFT step
    FFT_freq = np.fft.rfftfreq(N // tap) * fs

    return FFT_sum, FFT_freq
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift, fftfreq

def createComplexChirp(PulseLength, CentralPulseFrq, PulseBandWidth, SamplingRate):
    NumSamples = int(PulseLength * SamplingRate)
    t_s = np.arange(NumSamples) / SamplingRate
    x = np.exp(1j * 2.0 * np.pi * ((CentralPulseFrq - PulseBandWidth / 2.0) * t_s
                                 + (PulseBandWidth / (2.0 * PulseLength)) * (t_s**2)))
    return x, t_s

# Parâmetros
SamplingRate = 100*1e6
PulseLength = 200*1e-6
PulseBandWidth = 5*1e6
CentralPulseFrq = 10*1e6

# Chirp e matched filter
pulse, t_s = createComplexChirp(PulseLength, CentralPulseFrq, PulseBandWidth, SamplingRate)
matched_filter = np.conj(np.flip(pulse))
out = signal.convolve(pulse, matched_filter, mode='same')

NumSamples = len(pulse)
freq = fftshift(fftfreq(NumSamples, 1 / SamplingRate))

# --- PLOTS ---
fig = plt.figure(figsize=(11, 7))
grid = plt.GridSpec(3, 2, hspace=0.4)

# Chirp (tempo)
ax1 = fig.add_subplot(grid[0, 0])
ax1.plot(t_s[:200], np.real(pulse[:200]), label='Parte real')
ax1.set_title('Chirp (trecho inicial)')
ax1.set_xlabel('Tempo (s)')
ax1.grid(True)

# Chirp (espectro)
ax2 = fig.add_subplot(grid[1, 0])
dsp = np.abs(fftshift(np.fft.fft(pulse)))
dsp /= np.max(dsp)
ax2.plot(freq / 1e6, 20*np.log10(dsp))
ax2.set_title('Espectro do Chirp')
ax2.set_xlabel('Frequência (MHz)')
ax2.set_ylabel('Magnitude (dB)')
ax2.grid(True)

# Chirp (espectrograma)
ax3 = fig.add_subplot(grid[2, 0])
f, t, Sxx = signal.spectrogram(np.real(pulse), fs=SamplingRate, nperseg=256)
ax3.pcolormesh(t, f/1e6, 10*np.log10(Sxx), shading='gouraud')
ax3.set_title('Espectrograma do Chirp')
ax3.set_ylabel('Frequência (MHz)')
ax3.set_xlabel('Tempo (s)')

# Filtro casado (tempo)
ax4 = fig.add_subplot(grid[0, 1])
ax4.plot(t_s, np.abs(out))
ax4.set_title('Saída do Filtro Casado (magnitude)')
ax4.set_xlabel('Tempo (s)')
ax4.grid(True)

# Filtro casado (espectro)
ax5 = fig.add_subplot(grid[1, 1])
dsp2 = np.abs(fftshift(np.fft.fft(out)))
dsp2 /= np.max(dsp2)
ax5.plot(freq / 1e6, 20*np.log10(dsp2))
ax5.set_title('Espectro da Saída')
ax5.set_xlabel('Frequência (MHz)')
ax5.set_ylabel('Magnitude (dB)')
ax5.grid(True)

# Filtro casado (espectrograma)
ax6 = fig.add_subplot(grid[2, 1])
f, t, Sxx = signal.spectrogram(np.real(out), fs=SamplingRate, nperseg=256)
ax6.pcolormesh(t, f/1e6, 10*np.log10(Sxx), shading='gouraud')
ax6.set_title('Espectrograma da Saída')
ax6.set_ylabel('Frequência (MHz)')
ax6.set_xlabel('Tempo (s)')

plt.show()

corr = signal.correlate(pulse, pulse, mode='same')

plt.figure(figsize=(8, 4))
plt.plot(t_s, np.abs(corr))
plt.title('Autocorrelação do Chirp')
plt.xlabel('Amostras')
plt.grid(True)
plt.show()
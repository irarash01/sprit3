# michel/nasim/ali/arash
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, convolve, firwin
import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np

# Parameters for the sine wave
#fs = 500  # Sampling frequency in Hz
#duration = 5  # Duration in seconds
#frequency = 60  # Frequency of the sine wave in Hz
#t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time vector
# Create a sine wave with the specified frequency
#amplitude = 1  # Amplitude of the sine wave
#input_signal = amplitude * np.sin(2 * np.pi * frequency * t)



from scipy.signal import butter, filtfilt
import numpy as np
import pyedflib

# Load the EEG file
file_name = 'excerpt1.prep.edf'
file = pyedflib.EdfReader(file_name)

# Read signals from the file
main_signal = file.readSignal(0)
manual_annotation_1 = file.readSignal(2)
manual_annotation_2 = file.readSignal(2)


A_extended = np.repeat(manual_annotation_1, 10)


# Get signal parameters
duration = int(file.file_duration)
length = len(main_signal)
fs = file.getSampleFrequency(0)

# Time vector for plotting (seconds)
t = np.linspace(0, duration, length)

win_s= 140000
win_ss=141000

# Now let's apply the 2nd-order Butterworth bandpass filter (11-16 Hz)


# Nyquist frequency
nyquist = fs / 2

# Design a 2nd-order Butterworth bandpass filter (11-16 Hz)
low_cutoff = 11 / nyquist  # Normalized low cutoff frequency (11 Hz)
high_cutoff = 16 / nyquist  # Normalized high cutoff frequency (16 Hz)

# Create a 2nd-order Butterworth bandpass filter
b, a = butter(3, [low_cutoff, high_cutoff], btype='band')

# Apply the filter to the sine wave input_signal using filtfilt for zero-phase filtering
filtered_signal = filtfilt(b, a, main_signal)

# Plot the filtered signal
plt.figure(figsize=(12, 6))

plt.plot(t[win_s:win_ss], 0.2*main_signal[win_s:win_ss], linewidth=0.5, color='red')
plt.plot(t[win_s:win_ss], 6*A_extended[win_s:win_ss], linewidth=2, color='black')
plt.plot(t[win_s:win_ss], filtered_signal[win_s:win_ss], linewidth=0.5, color='blue')
plt.title('Filtered Sine Wave (2nd-order Bandpass 11-16 Hz)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.ylim([-20.5, 20.5])
plt.tight_layout()
plt.show()

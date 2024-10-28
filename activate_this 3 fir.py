import matplotlib.pyplot as plt
from scipy.signal import convolve, firwin
import numpy as np
import pyedflib

# Load the EEG file
file_name = 'excerpt1.prep.edf'
file = pyedflib.EdfReader(file_name)

# Read signals from the file
main_signal = file.readSignal(0)
manual_annotation_1 = file.readSignal(1)
manual_annotation_2 = file.readSignal(2)

ann_1_extended = np.repeat(manual_annotation_1, 10)
ann_2_extended = np.repeat(manual_annotation_2, 10)

# Get signal parameters
duration = int(file.file_duration)
length = len(main_signal)
fs = file.getSampleFrequency(0)

# Time vector for plotting (seconds)
t = np.linspace(0, duration, length)

win_s = 70000
win_ss = win_s + 5000

# Design a FIR bandpass filter (11-16 Hz)
lowcut = 11  # Lower cutoff frequency in Hz
highcut = 16  # Upper cutoff frequency in Hz

# Design the FIR filter using firwin
fir_coeff = firwin(numtaps=50, cutoff=[lowcut, highcut], pass_zero='bandpass', fs=fs)

# Apply the FIR filter using convolution
fir_filtered_signal = convolve(main_signal, fir_coeff, mode='same')

# Raise the filtered signal to the power of 2 (element-wise squaring)
squared_filtered_signal = fir_filtered_signal ** 2

maxx = max(main_signal)
mx2 = maxx / 3

# Plot the filtered signal
plt.figure(figsize=(12, 6))
plt.plot(t[win_s:win_ss], 0.2 * main_signal[win_s:win_ss], linewidth=0.5, color='red')
plt.plot(t[win_s:win_ss], 300 * ann_1_extended[win_s:win_ss], linewidth=1, color='black')
plt.plot(t[win_s:win_ss], 300 * ann_2_extended[win_s:win_ss], linewidth=1, color='black', linestyle="-.")

plt.plot(t[win_s:win_ss], squared_filtered_signal[win_s:win_ss], linewidth=0.5, color='blue')

# Threshold settings
threshold = mx2
min_duration_samples = int(0.03 * fs)  # Minimum duration in samples (0.5 seconds)

# Find segments where squared signal exceeds threshold
above_threshold = squared_filtered_signal[win_s:win_ss] > threshold
segments = []
i = 0

# Loop through the signal to find segments exceeding the threshold
while i < len(above_threshold):
    if above_threshold[i]:  # Start of a potential segment
        start = i
        while i < len(above_threshold) and above_threshold[i]:
            i += 1
        end = i  # End of the segment
        # Check if the segment length exceeds the minimum duration
        if (end - start) >= min_duration_samples:
            segments.append((start, end))
    else:
        i += 1

# Prepare points for plotting asterisks
high_values_times = []
high_values_amplitudes = []

for start, end in segments:
    high_values_times.extend(t[win_s + start:win_s + end])  # Time points
    high_values_amplitudes.extend(squared_filtered_signal[win_s + start:win_s + end])  # Amplitudes

# Plot asterisks at points above the threshold with sufficient duration
plt.scatter(high_values_times, high_values_amplitudes, color='black', marker='*', s=200, label='Amplitude > Threshold Duration')

plt.title('Filtered EEG Signal (11-16 Hz)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.ylim([-20.5, 420.5])
plt.legend()
plt.tight_layout()
plt.show()

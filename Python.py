import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import numpy as np
import pyedflib
import pywt

# Load the EEG file
file_name = 'excerpt1.prep.edf'  # Replace with your actual file
file = pyedflib.EdfReader(file_name)

# Read signals from the file
main_eeg = file.readSignal(0)
manual_annotation_1 = file.readSignal(1)
manual_annotation_2 = file.readSignal(2)
# Get signal parameters
duration = int(file.file_duration)
length = len(main_eeg)
fs = file.getSampleFrequency(0)

# the sample rate of annotations were 10 times lower
ann_1_extended = np.repeat(manual_annotation_1, fs/10)
ann_2_extended = np.repeat(manual_annotation_2, fs/10)
# Time vector for plotting (seconds)
t = np.linspace(0, duration, length)

# windowing
win_s= 0
win_ss=141500
eeg=main_eeg[win_s:win_ss]

print(len(main_eeg),fs)

# Spindle detection function with intermediate plotting
def detect_spindles(eeg, fs):
    # Parameters
    spindle_freq_range = (11, 16)  # Frequency range of spindles (Hz)
    wavelet_name = 'mexh'  # Mexican Hat Wavelet
    min_duration = 0.5  # Minimum duration of spindle (in seconds)

    # Step 1: Continuous Wavelet Transform (CWT)
    scales = np.arange(1, 128)  # Define scales for CWT
    coefficients, freqs = pywt.cwt(eeg, scales, wavelet_name, 1.0 / fs)

    # Plotting the wavelet coefficients
    plt.figure(figsize=(12, 4))
    plt.imshow(np.abs(coefficients), aspect='auto', extent=[t[0], t[-1], freqs[-1], freqs[0]])
    plt.title('Wavelet Coefficients (Scales)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='Coefficient Magnitude')
    plt.show()

    # Focus on the spindle frequency range (11-16 Hz)
    spindle_coeffs = np.abs(coefficients[(freqs >= spindle_freq_range[0]) & (freqs <= spindle_freq_range[1]), :])

    # Plot spindle frequency range coefficients
    plt.figure(figsize=(12, 4))
    plt.imshow(spindle_coeffs, aspect='auto', extent=[t[0], t[-1], spindle_freq_range[1], spindle_freq_range[0]])
    plt.title('Spindle Frequency Range Coefficients (11-16 Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='Coefficient Magnitude')
    plt.show()

    # Step 2: Sliding window to detect spindles (top 10% coefficients)
    binary_signal = np.zeros(eeg.shape)
    threshold = np.percentile(spindle_coeffs, 90)  # Top 10% of wavelet coefficients
    binary_signal[spindle_coeffs.max(axis=0) >= threshold] = 1

    # Plot the binary detection (spindle candidates)
    plt.figure(figsize=(12, 4))
    plt.plot(t, binary_signal, label='Binary Detection (Spindle Candidates)', color='red')
    plt.title('Binary Spindle Detection')
    plt.xlabel('Time (s)')
    plt.ylabel('Binary Signal')
    plt.show()

    # Step 3: Rectified Envelope Method for spindle validation
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    # Apply bandpass filter for spindle frequency range
    filtered_signal = bandpass_filter(eeg, spindle_freq_range[0], spindle_freq_range[1], fs)

    # Plot the filtered signal in the spindle range
    plt.figure(figsize=(12, 4))
    plt.plot(t, filtered_signal, label='Filtered Signal (11-16 Hz)', color='orange')
    plt.title('Filtered EEG in Spindle Frequency Range (11-16 Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    # Calculate the envelope of the filtered signal
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)

    # Low-pass filter the envelope to smooth it
    def lowpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, data)

    smoothed_envelope = lowpass_filter(envelope, 2, fs)  # 2 Hz low-pass filter

    # Plot the smoothed envelope
    plt.figure(figsize=(12, 4))
    plt.plot(t, smoothed_envelope, label='Smoothed Envelope', color='green')
    plt.title('Smoothed Envelope of the Filtered Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Envelope Amplitude')
    plt.show()

    # Step 4: Detect spindle events
    spindle_candidates = np.where(binary_signal == 1)[0]  # Indices of potential spindles
    spindles = []

    # Group continuous spindle candidate indices into events
    if len(spindle_candidates) > 0:
        current_spindle = [spindle_candidates[0]]
        for i in range(1, len(spindle_candidates)):
            if spindle_candidates[i] - spindle_candidates[i - 1] == 1:
                current_spindle.append(spindle_candidates[i])
            else:
                # If spindle duration is long enough, add it to the list
                if len(current_spindle) >= min_duration * fs:
                    start_time = current_spindle[0] / fs
                    duration = len(current_spindle) / fs
                    spindles.append([start_time, duration])
                current_spindle = [spindle_candidates[i]]

        # Add last spindle if it meets the criteria
        if len(current_spindle) >= min_duration * fs:
            start_time = current_spindle[0] / fs
            duration = len(current_spindle) / fs
            spindles.append([start_time, duration])

    return np.array(spindles), filtered_signal

# Plotting function for spindles
def plot_spindles(eeg, fs, spindles, filtered_signal):
    time = np.arange(0, len(eeg)) / fs

    plt.figure(figsize=(15, 8))

    # Plot original EEG signal
    plt.subplot(3, 1, 1)
    plt.plot(time, eeg, label='EEG Signal')
    plt.title('Original EEG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot filtered signal (Spindle range)
    plt.subplot(3, 1, 2)
    plt.plot(time, filtered_signal, label='Filtered Spindle Signal (11-16 Hz)', color='orange')
    plt.title('Filtered EEG (Spindle Frequency Range)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot spindles on top of the filtered signal
    plt.subplot(3, 1, 3)
    plt.plot(time, filtered_signal, label='Filtered Signal', color='orange')
    for spindle in spindles:
        start_time, duration = spindle
        plt.axvspan(start_time, start_time + duration, color='red', alpha=0.3, label='Detected Spindle')
    plt.title('Detected Spindles')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage of the script
if __name__ == "__main__":
    # Detect spindles
    spindles, filtered_signal = detect_spindles(eeg, fs)

    # Plot the results
    plot_spindles(eeg, fs, spindles, filtered_signal)

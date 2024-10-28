# michel/nasim/ali/arash
# BPS , Second sprint
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import numpy as np
import pyedflib
import pywt


def read_signal(file_name, win_s, win_width):
    # Load the EEG file (replace with the actual file path)

    file = pyedflib.EdfReader(file_name)

    # Read the EEG signal and manual annotations from the file
    main_eeg = file.readSignal(0)  # EEG signal
    manual_annotation_1 = file.readSignal(1)  # First manual annotation
    manual_annotation_2 = file.readSignal(2)  # Second manual annotation

    # Get signal duration and sampling rate
    duration = int(file.file_duration)
    length = len(main_eeg)
    fs = file.getSampleFrequency(0)

    # Upsample annotations (assuming annotation frequency is x-time lower than the EEG)
    ann_1_extended = np.repeat(manual_annotation_1, len(main_eeg) // len(manual_annotation_1))
    ann_2_extended = np.repeat(manual_annotation_2, len(main_eeg) // len(manual_annotation_1))

    # Define time vector for plotting (in seconds)
    # t = np.linspace(0, duration, length)

    # Define the window of interest
    #    win_s = 39000
    #    win_width = 1000

    win_ss = win_s + win_width
    eeg = main_eeg[win_s:win_ss]

    ann_1_win = ann_1_extended[win_s:win_ss]
    ann_2_win = ann_2_extended[win_s:win_ss]

    print(f'Length of EEG signal: {len(main_eeg)}, Sampling rate: {fs} Hz')
    print(f'Length of interested window EEG signal: {len(eeg)}')

    return eeg, fs, ann_1_win, ann_2_win


# Spindle detection function with enhanced method (SWPE-E)
def detect_spindles(eeg, fs):
    # Parameters
    spindle_freq_range = (11, 16)  # Frequency range of spindles (Hz)
    wavelet_name = 'mexh'  # Mexican Hat Wavelet for CWT
    min_duration = 0.03  # Minimum spindle duration in seconds

    # Step 1: Continuous Wavelet Transform (CWT) for EEG signal
    scales = np.arange(1, 129)  # Scales for CWT
    coefficients, freqs = pywt.cwt(eeg, scales, wavelet_name, 1.0 / fs)

    # Focus on the spindle frequency range (11-16 Hz)
    spindle_coeffs = np.abs(coefficients[(freqs >= spindle_freq_range[0]) & (freqs <= spindle_freq_range[1]), :])

    # Step 2: Sliding window to detect spindle candidates based on top 3% of wavelet coefficients
    binary_signal = np.zeros(eeg.shape)
    threshold = np.percentile(spindle_coeffs, 90)  # Top 3% of wavelet coefficients
    binary_signal[spindle_coeffs.max(axis=0) >= threshold] = 1

    # Plot the binary detection (spindle candidates)
    # plt.figure(figsize=(12, 4))
    # plt.plot(binary_signal, label='Binary Detection (Spindle Candidates)', color='red')
    # plt.plot(ann_1_win, color='blue')
    # plt.plot(ann_2_win, color='black')
    # plt.title('Binary Spindle Detection')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Binary Signal')
    # plt.show()

    # Step 3: Rectified Envelope Method for spindle validation (EXTRA)
    # Apply bandpass filter for spindle frequency range
    nyquist = 0.5 * fs
    low = spindle_freq_range[0] / nyquist
    high = spindle_freq_range[1] / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, eeg)

    # Calculate the envelope of the filtered signal
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)

    # Smooth the envelope with a low-pass filter
    cutoff = 2
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low')

    smoothed_envelope = filtfilt(b, a, envelope)  # Low-pass filter with 2 Hz cutoff

    # Step 4: Detect spindle events from candidates and envelope
    spindle_candidates = np.where(binary_signal == 1)[0]  # Indices of spindle candidates
    spindles = []  # List to store detected spindles

    min_samples = int(min_duration * fs)  # Convert minimum spindle duration to samples

    # Group continuous spindle candidate indices into spindle events
    if len(spindle_candidates) > 0:
        current_spindle = [spindle_candidates[0]]
        for i in range(1, len(spindle_candidates)):
            # If the current candidate is contiguous with the previous one
            if spindle_candidates[i] - spindle_candidates[i - 1] == 1:
                current_spindle.append(spindle_candidates[i])
            else:
                # If the duration of the spindle is long enough, store it
                if len(current_spindle) >= min_samples:
                    start_time = current_spindle[0] / fs  # Convert start sample to time (seconds)
                    duration = len(current_spindle) / fs  # Convert length to time (seconds)
                    spindles.append([start_time, duration])
                current_spindle = [spindle_candidates[i]]  # Start a new spindle

        # Handle the last spindle (if it meets the criteria)
        if len(current_spindle) >= min_samples:
            start_time = current_spindle[0] / fs
            duration = len(current_spindle) / fs
            spindles.append([start_time, duration])

    # Return the detected spindles and the filtered signal
    return np.array(spindles), filtered_signal, smoothed_envelope


# Function to calculate TP, TN, FP, FN, precision, and sensitivity
def calculate_metrics(detected_spindles, manual_annotation):
    TP, FP, FN, F1 = 0, 0, 0, 0

    for spindle in detected_spindles:
        start, duration = spindle
        spindle_time_indices = np.arange(int(start * fs), int((start + duration) * fs))
        if np.any(manual_annotation[spindle_time_indices] == 1):  # True Positive
            TP += 1
        else:  # False Positive
            FP += 1

    # Calculate False Negatives
    for spindle_index in np.where(manual_annotation == 1)[0]:
        if not np.any(
                [spindle_index in np.arange(int(sp[0] * fs), int((sp[0] + sp[1]) * fs)) for sp in detected_spindles]):
            FN += 1

    TN = len(manual_annotation) - (TP + FP + FN)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    try:
        F1 = 2 * precision * sensitivity / (precision + sensitivity)
    except ZeroDivisionError:
        F1 = 0

    return TP, TN, FP, FN, precision, sensitivity, F1


# Plotting function for spindles
def plot_spindles(eeg, fs, spindles, ann_1_win, filtered_eeg, envelope_eeg):


    # Generate the time axis in seconds for plotting
    time = np.arange(0, len(eeg)) / fs

    plt.figure(figsize=(15, 8))
    # Plot detected spindles on the filtered signal
    # plt.subplot(3, 1, 3)
    plt.plot(time, eeg, label='Orginal Signal', color='orange')
    plt.plot(time, ann_1_win * 60, label='annotate spindle', color='black')
    plt.plot(time, filtered_eeg, label='filtered Signal', color='red')
    plt.plot(time, envelope_eeg, label='envelope Signal', color='blue')

    # Highlight detected spindles
    for spindle in spindles:
        start_time, duration = spindle
        plt.axvspan(start_time, start_time + duration, color='black', alpha=0.3, label='Detected Spindle')

    plt.title('Detected Spindles')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Only show unique labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig('Final plot', dpi=600)  # Save the plot as a PNG file with 300 DPI
    plt.show()


# Example usage of the script
if __name__ == "__main__":
    eeg, fs, ann_1_win, ann_2_win = read_signal('excerpt1.prep.edf', win_s=60000, win_width=5000)
    # Detect spindles in EEG signal
    spindles, filtered_eeg, envelop_eeg = detect_spindles(eeg, fs)

    # Compare detected spindles with both manual annotations
    TP1, TN1, FP1, FN1, precision1, sensitivity1, F1 = calculate_metrics(spindles, ann_1_win)
    TP2, TN2, FP2, FN2, precision2, sensitivity2, F2 = calculate_metrics(spindles, ann_2_win)

    # Print metrics for manual annotation 1
    print(
        f'Metrics for Manual Annotation 1: \nTP: {TP1}, TN: {TN1},'
        f' FP: {FP1}, FN: {FN1}, Precision: {precision1:.3f}, Sensitivity: {sensitivity1:.3f}, F1 score {F1: 03f}')

    # Print metrics for manual annotation 2
    print(
        f'Metrics for Manual Annotation 2: \nTP: {TP2}, TN: {TN2},'
        f' FP: {FP2}, FN: {FN2}, Precision: {precision2:.3f}, Sensitivity: {sensitivity2:.3f}, F1 score {F2: 03f}')

    # Plot the detected spindles
    plot_spindles(eeg, fs, spindles, ann_1_win, filtered_eeg, envelop_eeg)

    plot_spindles(eeg, fs, spindles, ann_2_win, filtered_eeg, envelop_eeg)


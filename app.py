import pandas as pd
import numpy as np
import joblib
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load('model.pkl')

# Function to preprocess raw ECG data from CSV
def preprocess_ecg_csv(csv_path):
    # Load the raw ECG signal data
    ecg_data = pd.read_csv(csv_path)

    # Assume the columns are 'Time (ms)' for timestamps and 'ECG Signal' for the raw ECG data
    ecg_signal = ecg_data['ECG Signal'].values

    # Normalize the ECG signal (between -1 and 1 for example)
    normalized_signal = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal)) * 2 - 1

    # Plot the normalized ECG signal (optional, remove if not needed for visualization)
    plt.plot(normalized_signal)
    plt.title('Normalized ECG Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('ECG Amplitude')
    plt.show()

    # Detect R-peaks (assuming that R-peaks are the highest points in the signal)
    peaks, _ = find_peaks(normalized_signal, height=0.5)  # Adjust height as needed

    # Plot the ECG signal with R-peaks (optional, remove if not needed for visualization)
    plt.plot(normalized_signal)
    plt.plot(peaks, normalized_signal[peaks], "x")  # Mark the R-peaks with 'x'
    plt.title('ECG Signal with R-peaks')
    plt.xlabel('Time (ms)')
    plt.ylabel('ECG Amplitude')
    plt.show()

    # Calculate R-R intervals (differences between consecutive R-peaks)
    rr_intervals = np.diff(peaks)  # R-R intervals in milliseconds
    avg_heart_rate = 60 / np.mean(rr_intervals)  # Convert to heart rate (beats per minute)

    # Example of additional features:
    # Feature 1: Mean R-R interval
    feature1 = np.mean(rr_intervals)

    # Feature 2: Standard deviation of R-R intervals
    feature2 = np.std(rr_intervals)

    # Feature 3: Peak count (number of detected R-peaks)
    feature3 = len(peaks)

    # Feature 4: Frequency (mean heart rate)
    feature4 = avg_heart_rate

    # Combine features into a single array
    features = np.array([feature1, feature2, feature3, feature4]).reshape(1, -1)

    return features

# Example: New ECG CSV file to predict
csv_path = r"C:\Users\saira\OneDrive\Desktop\electrolyte_imbalance_projec\hyperkalemia_ecg_data.csv"  # Replace with the actual CSV file path

# Preprocess the ECG CSV data
X_new_processed = preprocess_ecg_csv(csv_path)

# Predict the abnormality using the trained model
prediction = model.predict(X_new_processed)

# Display the result
result = "Abnormal" if prediction == 1 else "Normal"
print(f"Prediction for the ECG signal: {result}")

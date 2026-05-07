import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate time data (in ms)
time = np.arange(0, 1000, 1)  # 1000ms, 1 second of ECG data

# Simulating a normal ECG signal with sinusoidal waves
# This is just a simple approximation and does not replicate a real ECG signal
normal_ecg = 0.5 * np.sin(2 * np.pi * 1 * time / 1000)  # 1 Hz (normal ECG frequency)

# Simulating a hyperkalemia signal (peaked T-wave)
# Introduce a large peak in the T-wave portion (typically after the QRS complex)
hyperkalemia_ecg = normal_ecg.copy()
hyperkalemia_ecg[600:650] += 1.2 * np.sin(2 * np.pi * 2 * (time[600:650] - 600) / 50)  # Peaked T-wave

# Combine the data into a DataFrame
df = pd.DataFrame({
    'Time (ms)': time,
    'ECG Signal': hyperkalemia_ecg
})

# Save to CSV
csv_file = r"C:\Users\saira\Downloads\hyperkalemia_ecg_data.csv"
df.to_csv(csv_file, index=False)

# Display the simulated ECG with a hyperkalemia abnormality
plt.plot(time, hyperkalemia_ecg)
plt.title("Simulated ECG Signal with Hyperkalemia (Peaked T-wave)")
plt.xlabel("Time (ms)")
plt.ylabel("ECG Signal Amplitude")
plt.show()

print(f"Simulated ECG data with hyperkalemia saved to: {csv_file}")

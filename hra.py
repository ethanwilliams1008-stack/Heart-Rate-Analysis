import streamlit as st
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

st.title("🫀 ECG Analysis Dashboard")

# -----------------------------
# Patient selection dropdown
# -----------------------------
patients = {
    "Patient 1": "100",
    "Patient 2": "101",
    "Patient 3": "102",
    "Patient 4": "103",
    "Patient 5": "104",
    "Patient 6": "105",
    "Patient 7": "106",
    "Patient 8": "107",
    "Patient 9": "108",
    "Patient 10": "109",
}

selected_patient = st.selectbox("Select Patient", list(patients.keys()))
record_id = patients[selected_patient]

st.info(f"Analyzing MIT-BIH record {record_id}")

# -----------------------------
# Load ALL data once
# -----------------------------
@st.cache_data
def load_all_patients():
    patient_ids = ["100","101","102","103","104","105","106","107","108","109"]
    data = {}
    
    for pid in patient_ids:
        record = wfdb.rdrecord(pid, pn_dir='mitdb')
        signal = record.p_signal[:2000, 0]
        fs = record.fs
        data[pid] = (signal, fs)
    
    return data

all_data = load_all_patients()

# ✅ FIX: get selected patient data
signal, fs = all_data[record_id]

st.write(f"Loaded dataset for {selected_patient} (Record {record_id}) ✔")

# -----------------------------
# Filter function
# -----------------------------
def bandpass(signal, fs):
    nyq = 0.5 * fs
    b, a = butter(2, [0.5/nyq, 40/nyq], btype='band')
    return filtfilt(b, a, signal)

# -----------------------------
# Run analysis button
# -----------------------------
if st.button("Run ECG Analysis"):

    st.subheader(f"{selected_patient} Heart Rate Data")

    filtered = bandpass(signal, fs)

    # Raw ECG
    fig1, ax1 = plt.subplots()
    time = np.arange(len(signal)) / fs
    ax1.plot(time[:2000], signal[:2000])
    ax1.set_title("Raw ECG Signal")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude (mV)")
    st.pyplot(fig1)

    st.caption(
    "This graph shows the raw electrical activity of the heart over time. Each heartbeat is made up of several parts:\n\n"
    "• **P wave** – a small bump that represents the upper chambers of the heart (atria) preparing to contract.\n\n"
    "• **QRS complex** – a large spike that represents the main pumping action of the heart (ventricles contracting).\n\n"
    "• **T wave** – a smaller wave after the spike that shows the heart resetting for the next beat.\n\n"
    "Because this is raw data, it may include noise and interference, making these features harder to see clearly."
    )

    # Filtered ECG
    # Filtered ECG
    fig2, ax2 = plt.subplots()

    time = np.arange(len(signal)) / fs
    N = len(signal)

    ax2.plot(time[:N], filtered[:N])
    ax2.set_title("Filtered ECG Signal")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude (normalized)")

    # Safe annotation
    if len(filtered) > 200 and not np.isnan(filtered).any():
        ax2.text(time[200], np.max(filtered), "QRS Spike", fontsize=10)

    st.pyplot(fig2)

    st.caption(
    "This is the ECG signal after filtering out noise. The key parts of each heartbeat are now easier to see:\n\n"
    "• The **QRS complex (large spike)** stands out clearly and is the most important feature for detecting heartbeats.\n\n"
    "• The **P wave (small bump before the spike)** and **T wave (wave after the spike)** may also be visible depending on signal quality.\n\n"
    "Filtering helps isolate true heart activity from background noise, improving accuracy in analysis."
    )

    # Peaks
    peaks, _ = find_peaks(
        filtered,
        distance=int(fs * 0.6),
        height=np.mean(filtered) + np.std(filtered)
    )

    fig3, ax3 = plt.subplots()
    time = np.arange(len(signal)) / fs
    ax3.plot(time[:2000], filtered[:2000])
    ax3.plot(time[peaks], filtered[peaks], "rx")
    ax3.set_title("R-Peak Detection")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Amplitude (normalized)")
    st.pyplot(fig3)

    st.caption(
    "This graph shows detected heartbeats. Each red 'X' marks an **R-peak**, which is the tallest point of the QRS complex.\n\n"
    "• The **QRS complex** represents the main contraction of the heart.\n\n"
    "• The **R-peak** is used because it is the easiest and most reliable point to detect.\n\n"
    "By measuring the time between these peaks, we can calculate heart rate and analyze heart rhythm."
)

    # Heart rate
    if len(peaks) > 1:
        rr = np.diff(peaks) / fs
        hr = 60 / rr
        avg_hr = np.mean(hr)
        st.caption(
        "This value shows the average heart rate in beats per minute (BPM).\n\n"
        "It is calculated by measuring the time between consecutive **R-peaks** (heartbeats). "
        "Shorter intervals mean a faster heart rate, while longer intervals indicate a slower heart rate."
        )

        st.metric("Average Heart Rate (BPM)", f"{avg_hr:.1f}")

        st.caption(
        "A typical resting heart rate for adults is between **60 and 100 BPM**.\n\n"
        "• Below 60 BPM may indicate **bradycardia (slow heart rate)**\n"
        "• Above 100 BPM may indicate **tachycardia (fast heart rate)**\n\n"
        "Abnormal values do not always mean a problem but may require further medical evaluation."
        )

        if avg_hr > 100 or avg_hr < 60:
            st.error("Abnormal ECG detected")
    else:
        st.warning("Not enough peaks detected to compute heart rate")
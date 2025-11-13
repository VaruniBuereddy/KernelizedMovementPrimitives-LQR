import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_torque_csv(path, dt=None):
    """Load torque CSV, handle missing or constant time."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    df = pd.read_csv(path)
    if 'time' not in df.columns:
        df.columns = ['time'] + [f'tau{i+1}' for i in range(df.shape[1]-1)]
    
    # Convert time and torques to numeric
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    for c in df.columns:
        if c.startswith('tau'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)

    # Fix constant or missing time
    time = df['time'].to_numpy()
    if np.allclose(time, time[0]):
        if dt is None:
            dt = 1.0 / 500.0  # assume 500 Hz if unknown
        time = np.arange(len(df)) * dt
        df['time'] = time

    return df


def plot_torque_comparison(csv_paths, labels, dt_stt=0.001):
    """Plot 7 subplots (joints), each showing 3 torque curves from different methods."""
    if len(csv_paths) != 3:
        raise ValueError("Need exactly 3 CSV paths (KMP, DMP, STT)")

    # Load all three datasets
    kmp = load_torque_csv(csv_paths[0])
    dmp = load_torque_csv(csv_paths[1])
    stt = load_torque_csv(csv_paths[2], dt=dt_stt)

    # Determine common time range
    t_min = max(kmp['time'].min(), dmp['time'].min(), stt['time'].min())
    t_max = min(kmp['time'].max(), dmp['time'].max(), stt['time'].max())
    print(f"Clipping all plots to common time range: {t_min:.3f}–{t_max:.3f}s")

    # Clip all three to common window
    kmp = kmp[(kmp['time'] >= t_min) & (kmp['time'] <= t_max)]
    dmp = dmp[(dmp['time'] >= t_min) & (dmp['time'] <= t_max)]
    stt = stt[(stt['time'] >= t_min) & (stt['time'] <= t_max)]

    # Setup 7 subplots (shared x)
    fig, axs = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
    torque_cols = [f'tau{i+1}' for i in range(7)]

    for i, col in enumerate(torque_cols):
        axs[i].plot(kmp['time'], kmp[col], 'g', label=labels[0], linewidth=1.5)
        axs[i].plot(dmp['time'], dmp[col], 'm', label=labels[1], linewidth=1.5)
        axs[i].plot(stt['time'], stt[col], 'b', label=labels[2], linewidth=1.5)
        axs[i].set_ylabel(f"τ{i+1} (Nm)")
        axs[i].grid(True)
        if i == 0:
            axs[i].legend(loc='upper right', fontsize=8)

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint Torques Comparison (KMP+LQR vs DMP+UMPC vs STT)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# === Example usage ===
if __name__ == "__main__":
    csv_kmp = "/home/iisc-hiro-lab-phd-2/Documents/KMP/C_results/lqr_franka_torques.csv"
    csv_dmp = "/home/iisc-hiro-lab-phd-2/Documents/DMP_LfD/C_results/lqr_franka_torques.csv"
    csv_stt = "/home/iisc-hiro-lab-phd-2/Documents/STT_outputs/c_tau_d_calc_record.csv"

    plot_torque_comparison(
        [csv_kmp, csv_dmp, csv_stt],
        labels=["KMP+LQR", "DMP+UMPC", "STT"],
        dt_stt=0.001  # STT logged at 1 kHz
    )

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# === FILE PATHS ===
nominal_path = "/home/iisc-hiro-lab-phd-2/ros2_ws/src/franka_example_controllers/scripts/square_demonstrated_joint_positions(t,j1-j7,x,y,z).csv"
stt_learned_path = "/home/iisc-hiro-lab-phd-2/ros2_ws/src/franka_example_controllers/scripts/square_learned_end_effector_positions(t(ignore),x,y,z).csv"
kmp_lqr_path = "/home/iisc-hiro-lab-phd-2/Documents/KMP/results/lqr_franka_output.csv"
dmp_umpc_path = "/home/iisc-hiro-lab-phd-2/Documents/DMP_LfD/results/umpc_franka_output.csv"

# === READ DATA ===
# Nominal: no header, last three columns are x, y, z
nominal = pd.read_csv(nominal_path, header=None)
nominal_xyz = nominal.iloc[:, -3:].to_numpy()

# Stt learned: no header, last three columns are x, y, z
stt = pd.read_csv(stt_learned_path, header=None)
stt_xyz = stt.iloc[:, -3:].to_numpy()

# KMP+LQR: has header
kmp = pd.read_csv(kmp_lqr_path)
kmp_xyz = kmp[['x', 'y', 'z']].to_numpy()

# DMP+UMPC: has header
dmp = pd.read_csv(dmp_umpc_path)
dmp_xyz = dmp[['x', 'y', 'z']].to_numpy()

# === PLOT ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory
ax.plot(nominal_xyz[:, 0], nominal_xyz[:, 1], nominal_xyz[:, 2], 'r', label='Nominal', linewidth=2)
ax.plot(stt_xyz[:, 0], stt_xyz[:, 1], stt_xyz[:, 2], 'b', label='STT Learned', linewidth=2)
ax.plot(kmp_xyz[:, 0], kmp_xyz[:, 1], kmp_xyz[:, 2], 'g', linestyle='--', label='KMP+LQR', linewidth=2)
ax.plot(dmp_xyz[:, 0], dmp_xyz[:, 1], dmp_xyz[:, 2], 'm', linestyle='--', label='DMP+UMPC', linewidth=2)

# === TUBE AROUND NOMINAL ===
# Approximate the tube by plotting several offset curves around the nominal path
tube_radius = 0.002  # adjust tube radius
num_points = 20
for theta in np.linspace(0, 2 * np.pi, num_points):
    dx = tube_radius * np.cos(theta)
    dy = tube_radius * np.sin(theta)
    ax.plot(nominal_xyz[:, 0] + dx, nominal_xyz[:, 1] + dy, nominal_xyz[:, 2],
            color='r', alpha=0.2, linewidth=1)

# === STYLE ===
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('End Effector Trajectories')
ax.legend()
ax.grid(True)
ax.view_init(elev=25, azim=135)

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np

def save_acc_csv(csv_path: str, save_path: str, ee_cols=(8, 9, 10)):
    """
    Load an end-effector trajectory CSV (with time and positions),
    compute accelerations (via double differentiation), and save only [t, ax, ay, az].

    Args:
        csv_path: input trajectory file
        save_path: output file to save results
        ee_cols: column indices (0-based) for x, y, z positions
    """
    # Load data
    df = pd.read_csv(csv_path, header=None)
    num_cols = df.shape[1]
    df.columns = ["t"] + [f"c{i}" for i in range(1, num_cols)]

    # Extract EE position columns
    ee_names = [df.columns[i] for i in ee_cols]
    t = df["t"].values.astype(float)

    # Create new dataframe with time and accelerations
    acc_df = pd.DataFrame({"t": t})

    # Compute accelerations for each position dimension
    for i, col in enumerate(ee_names):
        pos = df[col].values.astype(float)
        vel = np.gradient(pos, t)
        acc = np.gradient(vel, t)
        acc_df[f"a{i+1}"] = acc  # acceleration columns a1, a2, a3

    # Save to CSV
    acc_df.to_csv(save_path, index=False)
    print(f"✅ Saved accelerations to {save_path}")
    print(f"Columns saved: {list(acc_df.columns)}")

# Example usage
save_acc_csv(
    "/home/iisc-hiro-lab-phd-2/Documents/KMP/scripts/paper_roboface_demonstrated_joint_positions(t,j1-j7,x,y,z).csv",
    "trajectory_with_acc.csv",
    ee_cols=(8, 9, 10)
)

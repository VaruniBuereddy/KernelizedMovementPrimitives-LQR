import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt


def load_and_compute_velocities(csv_path: str) -> pd.DataFrame:
    # Correct: file is comma-separated
    df = pd.read_csv(csv_path, header=None, sep=",")
    
    df.columns = ["t"] + [f"s{i+1}" for i in range(df.shape[1]-1)]
    
    t = df["t"].values.astype(float)

    # compute derivatives for each state w.r.t time
    for col in df.columns[1:]:
        df[f"{col}_dot"] = np.gradient(df[col].values.astype(float), t)
    
    return df

def augment_trajectory_physically(df, n_aug=10, pos_noise_std=1e-3):
    """
    Generate augmented trajectories while keeping velocities consistent with positions.
    
    Parameters:
        df: DataFrame with columns [t, s1,...,sn, s1_dot,...,sn_dot]
        n_aug: number of augmented trajectories to generate
        pos_noise_std: std deviation of Gaussian noise added to positions
    
    Returns:
        df_aug: concatenated DataFrame with original + augmented trajectories
    """
    df_aug_list = [df.copy()]  # include original
    n_states = (df.shape[1] - 1) // 2  # number of positions

    t = df["t"].values.astype(float)

    for _ in range(n_aug):
        df_new = df.copy()
        # Add Gaussian noise only to positions
        for i in range(n_states):
            df_new[f"s{i+1}"] += np.random.normal(0, pos_noise_std, size=len(df))
        
        # Recompute velocities from new positions
        for i in range(n_states):
            df_new[f"s{i+1}_dot"] = np.gradient(df_new[f"s{i+1}"].values, t)

        df_aug_list.append(df_new)

    df_aug = pd.concat(df_aug_list, ignore_index=True)
    return df_aug


def prepare_gp_data(df: pd.DataFrame):
    """
    Prepare GP inputs and outputs for n states.
    Inputs: X = t (N,1)
    Outputs: Y = [s1...sn, s1_dot...sn_dot] (N, 2n)
    """
    X = df[["t"]].values
    state_cols = [c for c in df.columns if c != "t"]
    Y = df[state_cols].values
    return X, Y

def train_separate_gps(X, Y):
    """
    Train separate GPs for each output dimension.
    Returns list of models.
    """
    models = []
    for i in range(Y.shape[1]):
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp.fit(X, Y[:, i])
        models.append(gp)
    return models

def predict_gps(models, X_query):
    """
    Predict mean and variance for each GP model at query points.
    Returns mean (N, d) and var (N, d).
    """
    means, vars_ = [], []
    for gp in models:
        m, v = gp.predict(X_query, return_std=True)
        means.append(m)
        vars_.append(v**2)
    return np.stack(means, axis=1), np.stack(vars_, axis=1)

def compute_QR_from_gp(mu: np.ndarray, var: np.ndarray, df: pd.DataFrame):
    """
    Compute time-varying Q_t, R_t matrices from GP predictions.

    Parameters
    ----------
    mu : np.ndarray
        GP mean predictions, shape (N, n) where N = timesteps, n = states.
    var : np.ndarray
        GP variances, shape (N, n).
    df : pd.DataFrame
        DataFrame returned from load_and_compute_velocities (contains *_dot columns).

    Returns
    -------
    Q_list : list of np.ndarray
        List of Q_t matrices, one per timestep.
    R_list : list of np.ndarray
        List of R_t matrices, one per timestep.
    """
    n = mu.shape[1]
    N = mu.shape[0]

    Q_list, R_list = [], []

    # Extract velocity variances (approx. from df using gradient columns)
    vel_cols = [c for c in df.columns if c.endswith("_dot")]
    vel = df[vel_cols].values
    vel_var = np.var(vel, axis=0)  # crude approx; could also GP them

    for t in range(N):
        # state uncertainty
        q_diag = 1.0 / (var[t, :] + 1e-6)   # inverse variance weighting
        Q_t = np.diag(q_diag)

        # control uncertainty (using same scheme, but on velocity variance)
        r_diag = 1.0 / (vel_var + 1e-6)
        R_t = np.diag(r_diag)

        Q_list.append(Q_t)
        R_list.append(R_t)

    return Q_list, R_list

import numpy as np

def safe_QR_from_var_and_vel(var, vel, t_index,
                            eps_var=1e-6, scale_Q=1.0,
                            lambda_R=1.0, lambda_E=1e-3,
                            q_clip=(1e-4, 1e8), r_clip=(1e-6, 1e6)):
    """
    Safe construction of Q_t and R_t for timestep t_index.

    var: (N, 2n) GP predictive variances (stacked outputs: states then state-derivatives)
    vel: (N, n) demo velocities (actual numerically computed velocities)
    t_index: integer index of the timestep to compute matrices for

    Returns:
        Q_t: (2n, 2n) state-cost matrix (diagonal)
        R_t: (n, n) control-cost matrix (diagonal)
    """
    # Q: invert variance with floor and optional global scaling
    diag_var = np.maximum(var[t_index, :], eps_var)   # floor tiny variances
    q_diag = (1.0 / diag_var) * scale_Q
    q_diag = np.clip(q_diag, q_clip[0], q_clip[1])
    Q_t = np.diag(q_diag)

    # R: energy-based (paper uses squared velocities)
    vel_t = vel[t_index]  # shape (n,)
    E_t = np.diag(vel_t**2) + lambda_E * np.eye(len(vel_t))
    R_t_full = lambda_R * np.linalg.inv(E_t)
    # keep R diagonal (paper uses diagonal Rt); clip diagonals to reasonable range
    r_diag = np.clip(np.diag(R_t_full), r_clip[0], r_clip[1])
    R_t = np.diag(r_diag)

    return Q_t, R_t

import numpy as np
from scipy.linalg import expm, block_diag, solve_continuous_lyapunov

def discretize_double_integrator(n, dt):
    """
    Build continuous-time A,B for double integrator of n DoFs (positions+vels),
    then compute discrete-time Ad, Bd under zero-order hold with timestep dt.
    Returns A (2n x 2n), B (2n x n), Ad, Bd.
    """
    N = 2 * n
    A = np.zeros((N, N))
    # top-right block is I_n
    A[:n, n:] = np.eye(n)
    # bottom rows zeros (double integrator)
    B = np.zeros((N, n))
    B[n:, :] = np.eye(n)  # control enters accelerations

    # Discretize using matrix exponential for zero-order hold:
    # [Ad  Bd] = exp([[A, B],[0, 0]] * dt)
    M = np.zeros((N + n, N + n))
    M[:N, :N] = A
    M[:N, N:] = B
    # lower block zeros
    Mexp = expm(M * dt)
    Ad = Mexp[:N, :N]
    Bd = Mexp[:N, N:]
    return A, B, Ad, Bd

def solve_time_varying_discrete_lqr(Ad, Bd, Q_list, R_list):
    """
    Solve finite-horizon time-varying discrete LQR.
    Inputs:
      Ad: (N,N) discrete-time A
      Bd: (N,M) discrete-time B
      Q_list: list/array of length T of (N,N) state cost matrices
      R_list: list/array of length T of (M,M) control cost matrices
    Returns:
      K_list: list of length T of (M, N) feedback gain matrices,
              where the control applied at time t is u_t = K_t (xi_hat_t - xi_t).
      P_list: list of length T+1 of Riccati matrices (P_0..P_T)
    """
    T = len(Q_list)
    N = Ad.shape[0]
    M = Bd.shape[1]
    # Preallocate
    P_list = [np.zeros((N,N)) for _ in range(T+1)]
    K_list = [np.zeros((M,N)) for _ in range(T)]

    # Terminal condition
    P_list[T] = Q_list[T-1].copy()  # if you want a terminal cost, else use Q_T (here use last Q)
    # Backward recursion from t = T-1 down to 0
    for t in range(T-1, -1, -1):
        Q_t = Q_list[t]
        R_t = R_list[t]
        P_next = P_list[t+1]

        # compute gain: K = (R + B^T P_next B)^{-1} B^T P_next A
        S = R_t + Bd.T @ P_next @ Bd
        # ensure S is symmetric PD; add tiny regularization if needed
        eps = 1e-9
        S = 0.5*(S + S.T) + eps * np.eye(M)
        K_t = np.linalg.solve(S, Bd.T @ P_next @ Ad)  # shape (M,N)
        # store K (note: this K works with u = -K x in classic form.
        # We'll use u = K (xi_hat - xi) below, which is equivalent.)
        K_list[t] = K_t

        # update P: P = Q + A^T P_next (A - B K)
        P_t = Q_t + Ad.T @ P_next @ (Ad - Bd @ K_t)
        # symmetrize to avoid numerical asymmetry
        P_list[t] = 0.5 * (P_t + P_t.T)

    return K_list, P_list

def simulate_closed_loop(Ad, Bd, K_list, xi0, xi_hat_seq):
    """
    Simulate closed-loop trajectories using u_t = K_t (xi_hat_t - xi_t).
    Inputs:
      Ad, Bd : discrete system matrices
      K_list : list of (M,N) gains for each t (length T)
      xi0    : initial state (N,)
      xi_hat_seq: desired state sequence shape (T, N)
    Returns:
      xi_traj: (T+1, N) states (xi_0..xi_T)
      u_traj: (T, M) control commands applied (u_0..u_{T-1})
    """
    T = len(K_list)
    N = Ad.shape[0]
    M = Bd.shape[1]
    xi = xi0.copy()
    xi_traj = np.zeros((T+1, N))
    xi_traj[0] = xi
    u_traj = np.zeros((T, M))
    for t in range(T):
        xi_hat = xi_hat_seq[t]
        Kt = K_list[t]          # shape (M,N)
        u = Kt @ (xi_hat - xi)  # u_t = K_t (xi_hat - xi)
        # apply system
        xi = Ad @ xi + Bd @ u
        xi_traj[t+1] = xi
        u_traj[t] = u
    return xi_traj, u_traj

def plot_results(t, mu_seq, traj):
    n = mu_seq.shape[1]
    fig, axs = plt.subplots(n, 1, figsize=(8, 2*n), sharex=True)

    for i in range(n):
        axs[i].plot(t, mu_seq[:, i], "r--", label=f"GP mean s{i+1}")
        axs[i].plot(t, traj[:, i], "b-", label=f"LQR tracked s{i+1}")
        axs[i].set_ylabel(f"s{i+1}")
        axs[i].legend()

    axs[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()



# csv_path = "scripts/end_effector_positions.csv"
# df = load_and_compute_velocities(csv_path)
# print("Loaded data shape:", df.shape)
# print("Augmenting data...")
# df_aug = augment_trajectory_physically(df, n_aug=10, pos_noise_std=1e-3)
# aug_csv_path = "scripts/end_effector_positions_augmented.csv"
# df_aug.to_csv(aug_csv_path, index=False)
# print(f"Augmented trajectories saved to {aug_csv_path}")

# #print(df[["t"]].values.shape)  # should show (N,) and columns
# X, Y = prepare_gp_data(df_aug)  # X is (N,1) time, Y is (N,6)
# models = train_separate_gps(X, Y)
# mu, var = predict_gps(models, X)  # predictions on training inputs (for Q_t)
# print(len(mu), var.shape)

# print(f"variance min, mean and max: {var.min()}, {var.mean()}, {var.max()}")

# Q_list, R_list = compute_QR_from_gp(mu, var, df)

# # print("Q[0] =\n", Q_list[0])
# # print("R[0] =\n", R_list[0])
# # import numpy as np
# # print("cond(Q0):", np.linalg.cond(Q_list[0]))
# # print("cond(R0):", np.linalg.cond(R_list[0]))
# # print(f"Check for eigen values positive definiteness: {np.all(np.linalg.eigvals(Q_list[0]) > 0), np.all(np.linalg.eigvals(R_list[0]) > 0)}")


# dt = np.median(np.diff(df["t"].values))  # sampling time
# n = (Y.shape[1]) // 2   # number of states (pos only, no. of outputs /2 if including velocities)
# A, B, Ad, Bd = discretize_double_integrator(n, dt)

# K_list, P_list = solve_time_varying_discrete_lqr(Ad, Bd, Q_list, R_list)

# # Initial state (from data)
# x0 = np.concatenate([df[[f"s{i+1}" for i in range(n)]].iloc[0].values,
#                      df[[f"s{i+1}_dot" for i in range(n)]].iloc[0].values])

# simulate_closed_loop(Ad, Bd, K_list, x0, mu)
# xi_traj, u_traj = simulate_closed_loop(Ad, Bd, K_list, x0, mu)
# plot_results(df["t"].values, mu, xi_traj[:-1])


# -------------------------
# Data preparation
# -------------------------
csv_path = "scripts/end_effector_positions.csv"
df = load_and_compute_velocities(csv_path)
print("Loaded data shape:", df.shape)

# Augment data
print("Augmenting data...")
df_aug = augment_trajectory_physically(df, n_aug=10, pos_noise_std=1e-3)
aug_csv_path = "scripts/end_effector_positions_augmented.csv"
df_aug.to_csv(aug_csv_path, index=False)
print(f"Augmented trajectories saved to {aug_csv_path}")
print("Augmented data shape:", df_aug.shape)

# Prepare GP training data
X, Y = prepare_gp_data(df_aug)   # X: (4235,1), Y: (4235,6)



#  Save trained models 
import joblib
import os

# Train GP models
# print("Training GP models for each output dimension.")
# models = train_separate_gps(X, Y)

# # Save each trained GP model to disk
# for i, m in enumerate(models):
#     joblib.dump(m, f"gp_model_output{i+1}.pkl")
# print("GP models saved to disk.")

# Load trained GP models from disk
models = []
for i in range(Y.shape[1]):  # number of outputs
    fname = f"gp_model_output{i+1}.pkl"
    if os.path.exists(fname):
        m = joblib.load(fname)
        models.append(m)
    else:
        raise FileNotFoundError(f"{fname} not found, please train first.")
print("GP models loaded successfully.")

mu, var = predict_gps(models, X)  # predictions on all augmented data
print("GP prediction shapes:", mu.shape, var.shape)

print(f"variance min, mean and max: {var.min()}, {var.mean()}, {var.max()}")

# Build matching time vector for augmented data
# Repeat original times for each augmentation
t_aug = np.tile(df["t"].values, df_aug.shape[0] // df.shape[0])
assert len(t_aug) == len(mu), "time vector and GP output mismatch!"

# Compute Q and R
Q_list, R_list = compute_QR_from_gp(mu, var, df)

# Discretization
dt = np.median(np.diff(df["t"].values))  # original sampling time
n = (Y.shape[1]) // 2
A, B, Ad, Bd = discretize_double_integrator(n, dt)

K_list, P_list = solve_time_varying_discrete_lqr(Ad, Bd, Q_list, R_list)

# Initial state from first row of original data
x0 = np.concatenate([df[[f"s{i+1}" for i in range(n)]].iloc[0].values,
                     df[[f"s{i+1}_dot" for i in range(n)]].iloc[0].values])

# Simulation
xi_traj, u_traj = simulate_closed_loop(Ad, Bd, K_list, x0, mu)

# -------------------------
# Plotting
# -------------------------
def plot_results_augmented(t, mu, xi_traj, n):
    fig, axs = plt.subplots(n, 1, figsize=(10, 2*n))
    for i in range(n):
        axs[i].plot(t, mu[:len(t), i], "r.", markersize=2, alpha=0.3, label=f"GP mean s{i+1}")
        axs[i].plot(t, xi_traj[:len(t), i], "b-", label=f"LQR trajectory s{i+1}")
        axs[i].legend()
        axs[i].set_xlabel("time [s]")
        axs[i].set_ylabel(f"s{i+1}")
    plt.tight_layout()
    plt.show()

plot_results_augmented(df["t"].values, mu, xi_traj, n)
    
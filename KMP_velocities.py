import numpy as np
import pandas as pd
from scipy.linalg import block_diag, expm
from matplotlib import pyplot as plt

def load_and_compute_velocities(csv_path: str) -> pd.DataFrame:
    # Correct: file is comma-separated
    df = pd.read_csv(csv_path, header=None, sep=",")
    
    df.columns = ["t"] + [f"s{i+1}" for i in range(df.shape[1]-1)]
    
    t = df["t"].values.astype(float)

    # compute derivatives for each state w.r.t time
    for col in df.columns[1:]:
        df[f"{col}_dot"] = np.gradient(df[col].values.astype(float), t)
    
    return df

def generate_diverse_trajectories(df: pd.DataFrame, H: int, pos_range: float = 0.01, noise_std: float = 1):
    """
    Generate H synthetic trajectories from a single trajectory df.
    Each trajectory has a random mean shift within pos_range and small Gaussian noise.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original trajectory, must contain columns ['t', 's1', ..., 'sn'].
    H : int
        Number of synthetic trajectories to generate (including the original).
    pos_range : float
        Maximum absolute value for uniform random mean shift for each state.
    noise_std : float
        Standard deviation of additional Gaussian noise for positions.
    
    Returns
    -------
    df_all : pd.DataFrame
        Concatenated DataFrame of all trajectories with recomputed velocities.
        Columns: ['t', 's1', ..., 'sn', 's1_dot', ..., 'sn_dot']
    """
    df_all_list = []
    t = df["t"].values.astype(float)
    n_states = (df.shape[1] - 1)//2  # Exclude time and velocity columns
    print(f"n_states: {n_states}")
    for h in range(H):
        df_new = df.copy()
        # Random mean shift for each state
        mean_shifts = np.random.uniform(-pos_range, pos_range, size=n_states)
        for i in range(n_states):
            df_new[f"s{i+1}"] += mean_shifts[i]
            # Add small Gaussian noise
            df_new[f"s{i+1}"] += np.random.normal(0, noise_std, size=len(df))
            # Recompute velocities
            df_new[f"s{i+1}_dot"] = np.gradient(df_new[f"s{i+1}"].values, t)
        
        df_all_list.append(df_new)

    df_all = pd.concat(df_all_list, ignore_index=True)
    return df_all

import matplotlib.pyplot as plt

def plot_diverse_trajectories(df_diverse, n_states=3, H=20):
    """
    Plot all generated diverse trajectories in one figure per state.

    Parameters
    ----------
    df_diverse : pd.DataFrame
        DataFrame returned from generate_diverse_trajectories
    n_states : int
        Number of state dimensions
    H : int
        Number of trajectories
    """
    fig, axs = plt.subplots(n_states, 1, figsize=(10, 2.5*n_states), sharex=True)
    
    traj_len = df_diverse.shape[0] // H  # length of a single trajectory
    t = df_diverse["t"].values[:traj_len]  # time vector (assume all same)
    
    for i in range(n_states):
        for h in range(H):
            start_idx = h * traj_len
            end_idx = (h+1) * traj_len
            axs[i].plot(t, df_diverse[f"s{i+1}"].values[start_idx:end_idx], alpha=0.7)
        axs[i].set_ylabel(f"s{i+1}")
        axs[i].grid(True)
    
    axs[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()

def compute_trajectory_distribution(df_diverse: pd.DataFrame, H: int, T: int, n_states: int):
    """
    Compute mean and covariance trajectories from multiple diverse trajectories.

    Parameters
    ----------
    df_diverse : pd.DataFrame
        Concatenated diverse trajectories (H * T rows, 1 + n_states + n_states_dot columns)
    H : int
        Number of trajectories
    T : int
        Length of a single trajectory
    n_states : int
        Number of position states per timestep

    Returns
    -------
    mu : np.ndarray
        Mean trajectory, shape (T, n_states)
    Sigma : np.ndarray
        Covariance matrices at each timestep, shape (T, n_states, n_states)
    """

    # # Extract position columns
    # pos_cols = [f"s{i+1}" for i in range(n_states)]
    # data = df_diverse[pos_cols].values  # shape (H*T, n_states)
    pos_cols = [f"s{i+1}" for i in range(n_states)]
    vel_cols = [f"s{i+1}_dot" for i in range(n_states)]
    all_cols = pos_cols + vel_cols
    data = df_diverse[all_cols].values  # shape (H*T, 2*n_states)
    n_states = 2 * n_states  # positions + velocities

    # Reshape into (H, T, n_states)
    trajectories = data.reshape(H, T, n_states)

    # Preallocate
    mu = np.zeros((T, n_states))
    Sigma = np.zeros((T, n_states, n_states))

    # Compute mean and covariance per timestep
    for t in range(T):
        traj_t = trajectories[:, t, :]  # shape (H, n_states)
        mu[t] = np.mean(traj_t, axis=0)
        Sigma[t] = np.cov(traj_t, rowvar=False)  # shape (n_states, n_states)

    return mu, Sigma

def initialize_kmp_hyperparameters(n_states):
    """
    Define KMP hyperparameters for each state.

    Returns
    -------
    kmp_params : dict
        Contains sigma_f, l, lambda1, lambda2
    """
    kmp_params = {
        "sigma_f": 1.0,       # signal variance
        "l": 0.05,            # length-scale (smoothness)
        "lambda1": 1e-4,      # regularization for mean
        "lambda2": 1e-3       # regularization for covariance
    }
    return kmp_params




import numpy as np

def squared_exponential_kernel(X1, X2, sigma_f=1.0, l=0.05):
    """
    Compute SE kernel between two sets of inputs.
    X1: (N1, d)
    X2: (N2, d)
    returns (N1, N2)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    dists = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)
    K = sigma_f**2 * np.exp(-0.5 / l**2 * dists)
    return K

def kmp_predict(X_train, mu, Sigma, X_query, kmp_params):
    """
    Query KMP for mean and covariance at test points.
    
    X_train : (N, d) inputs used for trajectory distribution (time)
    mu      : (N, D) mean trajectory (positions + velocities)
    Sigma   : (N, D, D) covariance per timestep
    X_query : (M, d) new test inputs
    kmp_params: dict with sigma_f, l, lambda1, lambda2

    Returns:
        mu_star   : (M, D)
        Sigma_star: (M, D, D)
    """
    N, D = mu.shape
    M = X_query.shape[0]

    sigma_f = kmp_params["sigma_f"]
    l = kmp_params["l"]
    lambda1 = kmp_params["lambda1"]
    lambda2 = kmp_params["lambda2"]

    # Construct block kernel K (ND x ND)
    K_blocks = np.zeros((N*D, N*D))
    for i in range(N):
        for j in range(N):
            K_blocks[i*D:(i+1)*D, j*D:(j+1)*D] = squared_exponential_kernel(
                X_train[i:i+1], X_train[j:j+1], sigma_f, l) * np.eye(D)

    # Flatten mu and Sigma for block-wise computation
    mu_flat = mu.flatten()
    Sigma_block = block_diag(*Sigma)  # shape (N*D, N*D)

    # Compute k_* and k_**
    k_star = np.zeros((M*D, N*D))
    for i in range(M):
        for j in range(N):
            k_star[i*D:(i+1)*D, j*D:(j+1)*D] = squared_exponential_kernel(
                X_query[i:i+1], X_train[j:j+1], sigma_f, l) * np.eye(D)

    k_star_star = np.zeros((M*D, M*D))
    for i in range(M):
        k_star_star[i*D:(i+1)*D, i*D:(i+1)*D] = (sigma_f**2) * np.eye(D)

    # Mean prediction
    mu_star_flat = k_star @ np.linalg.solve(K_blocks + lambda1*Sigma_block, mu_flat)

    # Covariance prediction
    Sigma_star_flat = k_star_star - k_star @ np.linalg.solve(K_blocks + lambda2*Sigma_block, k_star.T)

    # Reshape to (M, D) and (M, D, D)
    mu_star = mu_star_flat.reshape(M, D)
    Sigma_star = np.zeros((M, D, D))
    for i in range(M):
        Sigma_star[i] = Sigma_star_flat[i*D:(i+1)*D, i*D:(i+1)*D]

    return mu_star, Sigma_star


def discretize_double_integrator(n_pos, dt):
    """
    Build discrete-time A, B matrices for a double integrator system
    with both positions and velocities in the state.

    State: [p; v]  (dimension = 2*n_pos)
    Input: u ~ acceleration (dimension = n_pos)

    Parameters
    ----------
    n_pos : int
        Number of position states (velocities are same dimension)
    dt : float
        Time step

    Returns
    -------
    Ad : (2n_pos x 2n_pos) ndarray
        Discrete-time state transition matrix
    Bd : (2n_pos x n_pos) ndarray
        Discrete-time input matrix
    """
    n = n_pos
    N = 2 * n  # total state size

    # Continuous-time dynamics
    A = np.zeros((N, N))
    A[:n, n:] = np.eye(n)       # p_dot = v
    # v_dot = u
    B = np.zeros((N, n))
    B[n:, :] = np.eye(n)

    # Discretization via matrix exponential
    M = np.zeros((N + n, N + n))
    M[:N, :N] = A
    M[:N, N:] = B
    Mexp = expm(M * dt)

    Ad = Mexp[:N, :N]
    Bd = Mexp[:N, N:]

    return Ad, Bd

def solve_time_varying_lqr(Ad, Bd, Q_list, R_list):
    """
    Solve finite-horizon time-varying discrete LQR.
    Returns list of feedback gains K_list.
    """
    T = len(Q_list)
    N = Ad.shape[0]
    M = Bd.shape[1]
    P_list = [np.zeros((N, N)) for _ in range(T + 1)]
    K_list = [np.zeros((M, N)) for _ in range(T)]

    # Terminal cost
    P_list[T] = Q_list[-1].copy()

    for t in range(T - 1, -1, -1):
        Q = Q_list[t]
        R = R_list[t]
        P_next = P_list[t + 1]

        S = R + Bd.T @ P_next @ Bd
        eps = 1e-9
        S = 0.5 * (S + S.T) + eps * np.eye(M)
        K_t = np.linalg.solve(S, Bd.T @ P_next @ Ad)
        K_list[t] = K_t

        P_t = Q + Ad.T @ P_next @ (Ad - Bd @ K_t)
        P_list[t] = 0.5 * (P_t + P_t.T)  # symmetrize

    return K_list, P_list

def simulate_closed_loop(Ad, Bd, K_list, xi0, xi_hat_seq):
    """
    Simulate closed-loop system: u_t = K_t (xi_hat_t - xi_t)
    Returns xi_traj (T+1 x N), u_traj (T x M)
    """
    T = len(K_list)
    N = Ad.shape[0]
    M = Bd.shape[1]

    xi = xi0.copy()
    xi_traj = np.zeros((T + 1, N))
    xi_traj[0] = xi
    u_traj = np.zeros((T, M))

    for t in range(T):
        xi_hat = xi_hat_seq[t]
        Kt = K_list[t]
        u = Kt @ (xi_hat - xi)
        xi = Ad @ xi + Bd @ u
        xi_traj[t + 1] = xi
        u_traj[t] = u

    return xi_traj, u_traj

def compute_QR_from_kmp(Sigma_star, R_scale=1e-2):
    """
    Compute Q_t = inv(Sigma_star) and R_t = scaled identity.
    
    Sigma_star: (T, 2n, 2n) predicted covariance matrices (positions + velocities)
    
    Returns
    -------
    Q_list : list of (2n, 2n) arrays
        State cost matrices
    R_list : list of (n, n) arrays
        Control cost matrices
    """
    T, state_dim, _ = Sigma_star.shape
    n = state_dim // 2  # number of position states
    Q_list = []
    R_list = []

    for t in range(T):
        # ensure positive definite
        Sigma_t = Sigma_star[t] + 1e-6 * np.eye(state_dim)
        Q_t = np.linalg.inv(Sigma_t)

        # Control cost (still only acts on accelerations, size n)
        R_t = R_scale * np.eye(n)

        Q_list.append(Q_t)
        R_list.append(R_t)

    return Q_list, R_list



csv_path = "scripts/end_effector_positions.csv"
df = load_and_compute_velocities(csv_path)

# Generate 20 diverse trajectories
df_diverse = generate_diverse_trajectories(df, H=20, pos_range=0.02, noise_std=2e-3)

print("Generated diverse trajectories shape:", df_diverse.shape)


# plot_diverse_trajectories(df_diverse, n_states=3, H=20)

mu, sigma = compute_trajectory_distribution(df_diverse, H=20, T=df.shape[0], n_states=3)

print("Mean trajectory shape:", mu.shape)
print("Covariance trajectory shape:", sigma.shape)

n_states = 3
kmp_params = initialize_kmp_hyperparameters(n_states)   


X_train = df_diverse[["t"]].values[:len(mu)]  # assume uniform time for each trajectory
X_query = X_train  # predicting at same points, could be new times
mu_star, Sigma_star = kmp_predict(X_train, mu, sigma, X_query, kmp_params)

print("Predicted mean shape:", mu_star.shape)
print("Predicted covariance shape:", Sigma_star.shape)


# n_states = mu_star.shape[1]
# t = X_query.flatten()

# plt.figure(figsize=(12, 6))
# for i in range(n_states):
#     mu_i = mu_star[:, i]
#     sigma_i = np.sqrt(Sigma_star[:, i, i])  # corrected
#     plt.plot(t, mu_i, label=f"s{i+1} mean")
#     plt.fill_between(t, mu_i - 2*sigma_i, mu_i + 2*sigma_i, alpha=0.3)
# plt.xlabel("time")
# plt.ylabel("state values")
# plt.title("KMP predicted trajectories with uncertainty")
# plt.legend()
# plt.show()

n_pos = mu_star.shape[1] // 2  # number of position states
n = 2 * n_pos  # total state size (positions + velocities)
dt = 0.01  # timestep
xi0 = mu_star[0]  # shape (6,)  # initial state [positions; velocities]

# 1. Discretize double integrator
Ad, Bd = discretize_double_integrator(n_pos, dt)

# 2. Compute Q and R from KMP covariance
Q_list, R_list = compute_QR_from_kmp(Sigma_star, R_scale=1e-2)

print("Ad shape:", Ad.shape)
print("Bd shape:", Bd.shape)
print("Q_list shape:", Q_list[0].shape, len(Q_list[0]))
print("R_list shape:", R_list[0].shape, len(R_list[0]))
# 3. Solve LQR
K_list, P_list = solve_time_varying_lqr(Ad, Bd, Q_list, R_list)

# 4. Prepare desired state sequence xi_hat (positions + velocities)
xi_hat_seq = mu_star#np.hstack([mu_star, np.zeros_like(mu_star)])

# 5. Simulate closed-loop trajectory
xi_traj, u_traj = simulate_closed_loop(Ad, Bd, K_list, xi0, xi_hat_seq)

print("Closed-loop trajectory shape:", xi_traj.shape)
print("Control commands shape:", u_traj.shape)

T = mu_star.shape[0]
n = mu_star.shape[1]
t = np.arange(T) * dt  # time vector

# Plot xi_traj vs KMP predictions
plt.figure(figsize=(12, 6))
for i in range(n//2):
    # KMP mean and uncertainty
    mu_i = mu_star[:, i]
    sigma_i = np.sqrt(Sigma_star[:, i, i])
    plt.plot(t, mu_i, 'r--', label=f"s{i+1} KMP mean" if i==0 else "")
    plt.fill_between(t, mu_i - 2*sigma_i, mu_i + 2*sigma_i, color='r', alpha=0.2)

    # Closed-loop trajectory
    xi_i_traj = xi_traj[1:, i]  # skip initial state xi0
    plt.plot(t, xi_i_traj, 'b', label=f"s{i+1} closed-loop" if i==0 else "")

plt.xlabel("Time (s)")
plt.ylabel("State values")
plt.title("Closed-loop trajectory tracking KMP predictions")
plt.legend()
plt.show()
# n = mu_star.shape[1] // 2  # number of position states
# T = mu_star.shape[0]
# t = np.arange(T) * dt  # time vector

# plt.figure(figsize=(12, 6))
# for i in range(n):
#     # KMP predicted velocity (assume last n outputs in mu_star are velocities)
#     mu_vel = mu_star[:, n + i]
#     sigma_vel = np.sqrt(Sigma_star[:, n + i, n + i])
#     plt.plot(t, mu_vel, 'r--', label=f"s{i+1} velocity mean" if i==0 else "")
#     plt.fill_between(t, mu_vel - 2*sigma_vel, mu_vel + 2*sigma_vel, color='r', alpha=0.2)

#     # Closed-loop velocity
#     vel_i_traj = xi_traj[1:, n + i]  # skip initial state
#     plt.plot(t, vel_i_traj, 'b', label=f"s{i+1} closed-loop velocity" if i==0 else "")

# plt.xlabel("Time (s)")
# plt.ylabel("Velocity values")
# plt.title("Closed-loop velocity tracking KMP predictions")
# plt.legend()
# plt.show()

import numpy as np
import pandas as pd
from scipy.linalg import block_diag, expm
from matplotlib import pyplot as plt

def load_and_compute_velocities(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, sep=",")
    df.columns = ["t"] + [f"s{i+1}" for i in range(df.shape[1]-1)]
    t = df["t"].values.astype(float)
    for col in df.columns[1:]:
        df[f"{col}_dot"] = np.gradient(df[col].values.astype(float), t)
    
    return df

def generate_diverse_trajectories(df: pd.DataFrame, H: int, pos_range: float = 0.01, noise_std: float = 1):
    df_all_list = []
    t = df["t"].values.astype(float)
    n_states = (df.shape[1] - 1)//2  # number of position states
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
    # Extract position columns
    pos_cols = [f"s{i+1}" for i in range(n_states)]
    data = df_diverse[pos_cols].values  # shape (H*T, n_states)

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
    kmp_params = {
        "sigma_f": 1.0,       # signal variance
        "l": 0.05,            # length-scale (smoothness)
        "lambda1": 1e-4,      # regularization for mean
        "lambda2": 1e-3       # regularization for covariance
    }
    return kmp_params




import numpy as np

def squared_exponential_kernel(X1, X2, sigma_f=1.0, l=0.05):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    dists = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)
    K = sigma_f**2 * np.exp(-0.5 / l**2 * dists)
    return K

def kmp_predict(X_train, mu, Sigma, X_query, kmp_params):
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


def discretize_double_integrator(n, dt):
    N = 2 * n
    A = np.zeros((N, N))
    A[:n, n:] = np.eye(n)
    B = np.zeros((N, n))
    B[n:, :] = np.eye(n)

    # Discretize using zero-order hold
    M = np.zeros((N + n, N + n))
    M[:N, :N] = A
    M[:N, N:] = B
    Mexp = expm(M * dt)
    Ad = Mexp[:N, :N]
    Bd = Mexp[:N, N:]
    return Ad, Bd

def solve_time_varying_lqr(Ad, Bd, Q_list, R_list):
    T = len(Q_list)
    N = Ad.shape[0]
    M = Bd.shape[1]
    P_list = [np.zeros((N, N)) for _ in range(T + 1)]
    K_list = [np.zeros((M, N)) for _ in range(T)]
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
    T, n, _ = Sigma_star.shape
    Q_list = []
    R_list = []
    for t in range(T):
        # ensure positive definite
        Sigma_t = Sigma_star[t] + 1e-6 * np.eye(n)
        Q_t = np.linalg.inv(Sigma_t)
        R_t = R_scale * np.eye(n)
        # Expand Q to double integrator size (2n x 2n)
        Q_full = np.block([
            [Q_t, np.zeros((n, n))],
            [np.zeros((n, n)), 0.01*Q_t]  # small weight on velocities
        ])
        Q_list.append(Q_full)
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

n = mu_star.shape[1]
dt = 0.01  # timestep
xi0 = np.zeros(2*n)  # initial state [positions; velocities]

# 1. Discretize double integrator
Ad, Bd = discretize_double_integrator(n, dt)

# 2. Compute Q and R from KMP covariance
Q_list, R_list = compute_QR_from_kmp(Sigma_star, R_scale=1e-2)

# 3. Solve LQR
K_list, P_list = solve_time_varying_lqr(Ad, Bd, Q_list, R_list)

# 4. Prepare desired state sequence xi_hat (positions + zero velocities)
xi_hat_seq = np.hstack([mu_star, np.zeros_like(mu_star)])

# 5. Simulate closed-loop trajectory
xi_traj, u_traj = simulate_closed_loop(Ad, Bd, K_list, xi0, xi_hat_seq)

print("Closed-loop trajectory shape:", xi_traj.shape)
print("Control commands shape:", u_traj.shape)

T = mu_star.shape[0]
n = mu_star.shape[1]
t = np.arange(T) * dt  # time vector

# Plot results
plt.figure(figsize=(12, 6))
for i in range(n):
    # KMP mean and uncertainty
    mu_i = mu_star[:, i]
    sigma_i = np.sqrt(Sigma_star[:, i, i])
    plt.plot(t, mu_i, 'r--', label=f"s{i+1} KMP mean" if i==0 else "")
    plt.fill_between(t, mu_i - 2*sigma_i, mu_i + 2*sigma_i, color='r', alpha=0.5)

    # Closed-loop trajectory
    xi_i_traj = xi_traj[1:, i]  # skip initial state xi0
    plt.plot(t, xi_i_traj, 'b', label=f"s{i+1} closed-loop" if i==0 else "")

plt.xlabel("Time (s)")
plt.ylabel("State values")
plt.title("Closed-loop trajectory tracking KMP predictions")
plt.legend()
plt.show()


# plt.figure(figsize=(12, 6))
# for i in range(n):
#     # KMP predicted velocity (assume last n outputs in mu_star are velocities)
#     mu_vel = mu_star[:, n + i]
#     sigma_vel = np.sqrt(np.diagonal(Sigma_star[:, n + i, n + i]))
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

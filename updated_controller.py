import numpy as np
from scipy.linalg import block_diag, expm
from initialize_KMP import *
import matplotlib.pyplot as plt

def squared_exponential_kernel(X1, X2, sigma_f=1.0, l=0.05):
    """
    Squared-Exponential (RBF) kernel between X1 and X2.
    X1: (N1, d)
    X2: (N2, d)
    returns (N1, N2)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    K = sigma_f**2 * np.exp(-0.5 / l**2 * dists)
    return K

def kmp_predict(Mu_ref, Sigma_ref, X_train, X_query, kmp_params):
    """
    Kernelized Movement Primitives (KMP) prediction.
    
    Parameters
    ----------
    Mu_ref : (T, D_out) mean trajectory from GMR
    Sigma_ref : (T, D_out, D_out) covariance trajectory from GMR
    X_train : (T, d) training input (e.g., time)
    X_query : (M, d) query input (e.g., new times)
    kmp_params : dict with keys ['sigma_f', 'l', 'lambda1', 'lambda2']
    
    Returns
    -------
    mu_star : (M, D_out) predicted mean
    Sigma_star : (M, D_out, D_out) predicted covariance
    """
    sigma_f = kmp_params.get("sigma_f", 1.0)
    l = kmp_params.get("l", 0.05)
    lambda1 = kmp_params.get("lambda1", 1e-4)
    lambda2 = kmp_params.get("lambda2", 1e-3)
    
    T, D_out = Mu_ref.shape
    M = X_query.shape[0]

    # 1) Flatten trajectory
    mu_flat = Mu_ref.flatten()  # shape (T*D_out,)
    Sigma_block = block_diag(*[Sigma_ref[t] for t in range(T)])  # shape (T*D_out, T*D_out)

    # 2) Construct block kernel K (T*D_out x T*D_out)
    K_blocks = np.zeros((T*D_out, T*D_out))
    for i in range(T):
        for j in range(T):
            K_ij = squared_exponential_kernel(X_train[i:i+1], X_train[j:j+1], sigma_f, l) * np.eye(D_out)
            K_blocks[i*D_out:(i+1)*D_out, j*D_out:(j+1)*D_out] = K_ij

    # 3) Construct k_star and k_star_star for queries
    k_star = np.zeros((M*D_out, T*D_out))
    k_star_star = np.zeros((M*D_out, M*D_out))
    for i in range(M):
        for j in range(T):
            k_ij = squared_exponential_kernel(X_query[i:i+1], X_train[j:j+1], sigma_f, l) * np.eye(D_out)
            k_star[i*D_out:(i+1)*D_out, j*D_out:(j+1)*D_out] = k_ij
        k_star_star[i*D_out:(i+1)*D_out, i*D_out:(i+1)*D_out] = sigma_f**2 * np.eye(D_out)

    # 4) Mean prediction
    mu_star_flat = k_star @ np.linalg.solve(K_blocks + lambda1 * Sigma_block, mu_flat)

    # 5) Covariance prediction
    Sigma_star_flat = k_star_star - k_star @ np.linalg.solve(K_blocks + lambda2 * Sigma_block, k_star.T)

    # 6) Reshape to (M, D_out) and (M, D_out, D_out)
    mu_star = mu_star_flat.reshape(M, D_out)
    Sigma_star = np.zeros((M, D_out, D_out))
    for i in range(M):
        Sigma_star[i] = Sigma_star_flat[i*D_out:(i+1)*D_out, i*D_out:(i+1)*D_out]

    return mu_star, Sigma_star

def plot_kmp_results(t_train, Mu_ref, Sigma_ref, t_query, mu_star, Sigma_star, n_states):
    fig, axs = plt.subplots(n_states, 1, figsize=(10, 3*n_states), sharex=True)

    for i in range(n_states):
        # Reference mean + variance
        ref_mean = Mu_ref[:, i]
        ref_std = np.sqrt(Sigma_ref[:, i, i])

        # KMP mean + variance
        pred_mean = mu_star[:, i]
        pred_std = np.sqrt(Sigma_star[:, i, i])

        # Plot reference trajectory
        axs[i].plot(t_train, ref_mean, "b-", label="GMR reference mean")
        axs[i].fill_between(
            t_train, ref_mean - 2*ref_std, ref_mean + 2*ref_std,
            color="b", alpha=0.2, label="Ref ±2σ" if i == 0 else None
        )

        # Plot KMP prediction
        axs[i].plot(t_query, pred_mean, "r--", label="KMP prediction mean")
        axs[i].fill_between(
            t_query, pred_mean - 2*pred_std, pred_mean + 2*pred_std,
            color="r", alpha=0.2, label="KMP ±2σ" if i == 0 else None
        )

        axs[i].set_ylabel(f"pos[{i}]")
        axs[i].grid(True)

    axs[-1].set_xlabel("time [s]")
    axs[0].legend()
    plt.tight_layout()
    plt.show()

def auto_kmp_hyperparams(df, Sigma_ref, factor_l=3.0, sigma_f_scale=1.0):
    # median spacing in training timestamps
    t_train = df["t"].values
    dt_med = np.median(np.diff(t_train))
    l = factor_l * dt_med
    # set sigma_f to match scale of Mu_ref variance (roughly)
    ref_diag = np.array([np.diag(Sigma_ref[t]) for t in range(Sigma_ref.shape[0])])  # (T, D)
    # typical variance magnitude across output dims
    var_mean = np.mean(ref_diag)
    # set sigma_f relative to this
    sigma_f = max(1e-6, np.sqrt(sigma_f_scale * var_mean))
    return l, sigma_f


def compute_QR_from_kmp(Sigma_star, R_scale=1e-2, w_vel=0.3, eps=1e-6):
    """
    Sigma_star: (T, 6, 6) over [pos(3), vel(3)]
    Returns Q_list (T x 6x6), R_list (T x 3x3) for double integrator on 3 pos dims.
    """
    T, n, _ = Sigma_star.shape
    assert n == 6, "Expected 6D output [pos(3), vel(3)]"

    Q_list, R_list = [], []
    for t in range(T):
        S = Sigma_star[t]
        # regularize & invert
        S = 0.5*(S + S.T) + eps*np.eye(6)
        S_inv = np.linalg.inv(S)

        # Optional: downweight velocity block inside Q
        # Extract blocks to scale velocities
        Q_pospos = S_inv[:3, :3]
        Q_posvel = S_inv[:3, 3:]
        Q_velpos = S_inv[3:, :3]
        Q_velvel = S_inv[3:, 3:] * w_vel

        Q_t = np.block([
            [Q_pospos, Q_posvel],
            [Q_velpos, Q_velvel]
        ])
        # symmetrize
        Q_t = 0.5*(Q_t + Q_t.T)
        Q_list.append(Q_t)

        # R corresponds to inputs (accelerations) of size 3
        R_t = R_scale * np.eye(3)
        R_list.append(R_t)
    return Q_list, R_list


def discretize_double_integrator(n, dt):
    """
    Discretize a double integrator system:
        x_dot = v
        v_dot = u
    State vector xi = [positions; velocities], size n (already positions+velocities)
    Input u = accelerations, size n/2 (number of position dimensions)

    Returns:
        Ad: (n x n) discrete-time state matrix
        Bd: (n x n/2) discrete-time input matrix
    """
    n_pos = n // 2  # number of position dimensions
    A = np.zeros((n, n))
    B = np.zeros((n, n_pos))

    # x_dot = v
    A[:n_pos, n_pos:] = np.eye(n_pos)
    # v_dot = u
    B[n_pos:, :] = np.eye(n_pos)

    # Discretize using zero-order hold
    M = np.zeros((n + n_pos, n + n_pos))
    M[:n, :n] = A
    M[:n, n:] = B
    Mexp = expm(M * dt)
    Ad = Mexp[:n, :n]
    Bd = Mexp[:n, n:]
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

def simulate_closed_loop(Ad, Bd, K_list, xi0, xi_hat_seq, u_ff_seq=None):
    T = len(K_list)
    N = Ad.shape[0]
    M = Bd.shape[1]
    assert xi_hat_seq.shape[0] >= T, "Need at least T reference states"

    xi = xi0.copy()
    xi_traj = np.zeros((T + 1, N))
    u_traj = np.zeros((T, M))
    xi_traj[0] = xi

    for t in range(T):
        xi_hat = xi_hat_seq[t]
        Kt = K_list[t]
        u_fb = Kt @ (xi_hat - xi)
        u_ff = u_ff_seq[t] if (u_ff_seq is not None) else  np.zeros(M)
        u = u_ff + u_fb
        xi = Ad @ xi + Bd @ u
        xi_traj[t+1] = xi
        u_traj[t] = u
    return xi_traj, u_traj


def run_controller(xi0=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    csv_path = "/home/iisc-hiro-lab-phd-2/Documents/data/c_1.csv"
    H = 10
    pos_range = 0.02
    noise_std = 2e-3
    n_states = 3
    nbStates = 10
    time_scale_range = (1,1)

    # 1) Load demonstration data and compute velocities
    df = load_and_compute_velocities(csv_path)
    demo_dt = np.median(np.diff(df["t"].values))
    print("Loaded demo, timesteps:", df.shape[0], "dt:", demo_dt)

    # 2) Generate synthetic diverse trajectories
    df_diverse = generate_diverse_trajectories_v2(
        df, H=H, pos_range=pos_range, noise_std=noise_std, time_scale_range=time_scale_range
    )

    # 3) Prepare data for GMM
    Data_mat, data_sklearn = prepare_data(df_diverse, H=H, T=df.shape[0], n_states=n_states)

    # 4) Fit GMM on [time, pos, vel]
    gmm = fit_gmm(data_sklearn, n_components=nbStates)

    # 5) Compute reference trajectory via GMR
    t_query = np.linspace(df["t"].min(), df["t"].max(), 2 * len(df))
    Mu_ref, Sigma_ref = compute_reference_trajectory(gmm, time_query=t_query, in_idx=[0], out_idx=None)
    print("Mu_ref shape:", Mu_ref.shape, "Sigma_ref shape:", Sigma_ref.shape)

    # 6) Run KMP on uniform time grid
    X_train = np.linspace(df["t"].min(), df["t"].max(), 2 * len(df))
    X_star = np.linspace(X_train.min(), X_train.max(), 3 * len(df)).reshape(-1, 1)
    kmp_params = {"sigma_f": 0.1, "l": 0.1, "lambda1": 1e-4, "lambda2": 1e-3}

    mu_star, Sigma_star = kmp_predict(Mu_ref, Sigma_ref, X_train, X_star, kmp_params)
    print("KMP prediction shape:", mu_star.shape, "Sigma_star shape:", Sigma_star.shape)

    # 7) Setup LQR
    T = mu_star.shape[0]
    t_min, t_max = X_star[0, 0], X_star[-1, 0]
    dt = (t_max - t_min) / (T - 1)

    pos_ref = mu_star[:, :3]
    vel_ref = mu_star[:, 3:]
    acc_ref = np.gradient(vel_ref, dt, axis=0)

    Ad, Bd = discretize_double_integrator(n=6, dt=dt)
    Q_list, R_list = compute_QR_from_kmp(Sigma_star, R_scale=1e-2, w_vel=0.3)
    K_list, _ = solve_time_varying_lqr(Ad, Bd, Q_list, R_list)

    xi_hat_seq = np.hstack([pos_ref, vel_ref])
    if xi0 is None:
        xi0 = xi_hat_seq[0]

    xi_traj, u_traj = simulate_closed_loop(Ad, Bd, K_list, xi0, xi_hat_seq, u_ff_seq=acc_ref)

    # 8) Plot results
    # t = np.arange(T) * dt
    # Time vectors
    T = xi_hat_seq.shape[0]
    dt = (X_star[-1,0] - X_star[0,0]) / (T - 1)
    t_ref = np.arange(T) * dt          # for desired (reference)
    t_act = np.arange(T + 1) * dt      # for actual (simulated)

    labels = ['x','y','z']

    # --- 1. Positions ---
    plt.figure(figsize=(10,6))
    for i in range(3):
        plt.plot(t_ref, pos_ref[:, i], 'k--', label=f'{labels[i]} desired' if i==0 else "")
        plt.plot(t_act, xi_traj[:, i], label=f'{labels[i]} actual')
    plt.title("Cartesian Positions: Desired vs Actual")
    plt.xlabel("Time [s]"); plt.ylabel("Position [m]")
    plt.legend(); plt.grid(True)

    # --- 2. Velocities ---
    plt.figure(figsize=(10,6))
    for i in range(3):
        plt.plot(t_ref, vel_ref[:, i], 'k--', label=f'{labels[i]}̇ desired' if i==0 else "")
        plt.plot(t_act, xi_traj[:, 3+i], label=f'{labels[i]}̇ actual')
    plt.title("Cartesian Velocities: Desired vs Actual")
    plt.xlabel("Time [s]"); plt.ylabel("Velocity [m/s]")
    plt.legend(); plt.grid(True)

    # --- 3. Accelerations ---
    plt.figure(figsize=(10,6))
    for i in range(3):
        plt.plot(t_ref, u_traj[:, i], label=f'u{i+1} (acc {labels[i]})')
    plt.title("Optimal Control Inputs (Accelerations)")
    plt.xlabel("Time [s]"); plt.ylabel("Acceleration [m/s²]")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 9) Save results
    save_dir = "/home/iisc-hiro-lab-phd-2/Documents/KMP/C_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lqr_kmp_results.npz")

    np.savez(save_path,
            t_ref=t_ref,
            t_act=t_act,
            xi_ref=xi_hat_seq,
            xi_act=xi_traj,
            u_traj=u_traj,
            acc_ref=acc_ref)
    print(f"Saved results to {save_path}")

    # ---------- New: CSV export ----------
    csv_path = os.path.join(save_dir, "lqr_kmp_results.csv")

    # Combine time, states, and accelerations into one table
    # xi_ref = [x, y, z, xdot, ydot, zdot]
    data_csv = np.hstack([t_ref.reshape(-1, 1), xi_hat_seq[:, :3], xi_hat_seq[:, 3:], acc_ref])

    header = "t,x,y,z,xdot,ydot,zdot,xddot,yddot,zddot"
    np.savetxt(csv_path, data_csv, delimiter=",", header=header, comments='')

    print(f"Saved readable trajectory to {csv_path}")

    return u_traj


if __name__ == "__main__":

    u_traj = run_controller()
    # np.save("u_traj.npy", u_traj)
    print("Control trajectory shape:", u_traj.shape)
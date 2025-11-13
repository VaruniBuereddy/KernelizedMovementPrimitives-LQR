import numpy as np
from scipy.linalg import block_diag
from initialize_KMP import *
import matplotlib.pyplot as plt
from scipy.linalg import block_diag, expm

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


# -------------------------

    # import scipy.io

    # mat = scipy.io.loadmat('/home/iisc-hiro-lab-phd-2/Documents/robInfLib-matlab/2Dletters/G.mat')
    # demos = mat['demos']

    # # demos is (1, 13), so access first cell:
    # first_demo = demos[0, 0]  # first element
    # print(type(first_demo))
    # print(first_demo)  # may be another array or dict depending on MATLAB structure

    # for i in range(demos.shape[1]):
    #     demo = demos[0, i]  # get i-th demo
    #     print(f"Demo {i}: type {type(demo)}, shape {demo.shape if hasattr(demo, 'shape') else demo}")

    # import pdb; pdb.set_trace()

    # 1) load single demo and compute velocities

if __name__ == "__main__":
    # ---- user parameters ----
    csv_path = "scripts/end_effector_positions.csv"  # change to your file
    H = 4                 # number of synthetic demos to generate
    pos_range = 0.02       # random mean shift range (meters)
    noise_std = 2e-3       # additive Gaussian noise std
    n_states = 3           # number of position dims in the demo
    nbStates = 10           # number of GMM components
    time_scale_range=(1,1)#(0.9, 1.1)
    
    df = load_and_compute_velocities(csv_path)
    T = df.shape[0]
    demo_dt = np.median(np.diff(df["t"].values))
    print("Loaded demo, timesteps:", T, "dt:", demo_dt)

    # 2) generate diverse synthetic trajectories
    df_diverse = generate_diverse_trajectories_v2(df, H=H, pos_range=pos_range, noise_std=noise_std, time_scale_range=time_scale_range)
    print("Generated diverse trajectories shape:", df_diverse.shape)
    plot_diverse_trajectories_time_scaled(df_diverse, H=H, n_states=n_states)

    # 3) prepare data (MATLAB-like and sklearn-like)
    Data_mat, data_sklearn = prepare_data(df_diverse, H=H, T=T, n_states=n_states)
    print("Data_mat shape (D x H*T):", Data_mat.shape, "data_sklearn shape (N_samples x D):", data_sklearn.shape)

    # 4) fit GMM on joint [time, pos..., vel...]
    print("Fitting GMM with", nbStates, "components ... (this may take a bit)")
    gmm = fit_gmm(data_sklearn, n_components=nbStates)
    print("GMM fitted. Means shape:", gmm.means_.shape, "Covariances shape:", gmm.covariances_.shape)

    # 5) compute reference trajectory via GMR at the original demo times
    t_query = np.linspace(df["t"].min(), df["t"].max(), 2*len(df))  
    Mu_ref, Sigma_ref = compute_reference_trajectory(gmm, time_query=t_query, in_idx=[0], out_idx=None)
    print()
    print("Mu_ref shape:", Mu_ref.shape)        # (T, D_out)
    print("Sigma_ref shape:", Sigma_ref.shape)  # (T, D_out, D_out)
    # import pdb; pdb.set_trace()
    kmp_params = {"sigma_f": 0.1, "l": 0.05, "lambda1": 1e-4, "lambda2": 1e-3}
    # l, sigma_f = auto_kmp_hyperparams(df, Sigma_ref, factor_l=3.0, sigma_f_scale=1.0)
    # kmp_params = {"sigma_f": sigma_f, "l": l, "lambda1": 1e-4, "lambda2": 1e-3}
    print("KMP hyperparams:", kmp_params)

    X_train = np.linspace(df["t"].min(), df["t"].max(), 2*len(df))#df_diverse["t"].values.reshape(-1, 1)  # shape (T,1)
    print("Last time: ", df['t'].max())
    # X_query = np.linspace(0, df_diverse['t'].values[-1], 2*df['t'].shape[0]).reshape(-1, 1)  # shape (M,1), more points for smoothness

    # mu_star, Sigma_star = kmp_predict(Mu_ref, Sigma_ref, X_train, X_query, kmp_params)

    # print("Predicted mean shape:", mu_star.shape)        # (M, D_out)
    # print("Predicted covariance shape:", Sigma_star.shape)  # (M, D_out, D_out)
    # print("predicted cov[0]:", Sigma_star[0])
    # plot_kmp_results(
    #     t_train=X_train.flatten(),
    #     Mu_ref=Mu_ref,
    #     Sigma_ref=Sigma_ref,
    #     t_query=X_query.flatten(),
    #     mu_star=mu_star,
    #     Sigma_star=Sigma_star,
    #     n_states=n_states
    # )

    t_min, t_max = X_train.min(), X_train.max()
    print(t_min, t_max)
    # print(df['t'].shape)
    X_far = np.linspace(t_min, t_max, 3*T).reshape(-1,1)
    # print("Extrapolation query times:", X_far.flatten())
    mu_far, Sigma_far = kmp_predict(Mu_ref, Sigma_ref, X_train, X_far, kmp_params)
    plot_kmp_results(
        t_train=X_train.flatten(),
        Mu_ref=Mu_ref,
        Sigma_ref=Sigma_ref,
        t_query=X_far.flatten(),
        mu_star=mu_far,
        Sigma_star=Sigma_far,
        n_states=n_states
    )

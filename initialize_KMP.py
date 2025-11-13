#!/usr/bin/env python3
"""
GMM+GMR pipeline for KMP initialization (positions + velocities).
Functions:
 - prepare_data(df_diverse, H, T, n_states)
 - fit_gmm(data, n_components=8)
 - gmr(gmm, X_query, in_idx=[0], out_idx=None)
 - compute_reference_trajectory(gmm, nbData, dt, in_idx=[0], out_idx=None)
Example usage provided in __main__.
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


# def load_and_compute_velocities(csv_path: str) -> pd.DataFrame:
#     """
#     Load CSV with columns [t, s1, s2, ...]. Compute numeric velocities and return df.
#     """
#     df = pd.read_csv(csv_path, header=None, sep=",")
#     df.columns = ["t"] + [f"s{i+1}" for i in range(df.shape[1]-1)]
#     t = df["t"].values.astype(float)
#     # compute derivatives for each state w.r.t time
#     for col in df.columns[1:]:
#         df[f"{col}_dot"] = np.gradient(df[col].values.astype(float), t)
#     return df

# def load_and_compute_velocities(csv_path: str, ee_cols=(8, 9, 10)):
#     # Load without headers
#     df = pd.read_csv(csv_path, header=None)

#     # Rename all columns
#     num_cols = df.shape[1]
#     df.columns = ["t"] + [f"c{i}" for i in range(1, num_cols)]

#     # Select x,y,z columns (by index)
#     # Add +1 because col0 is time, so actual df index shifts by 1.
#     ee_names = [df.columns[i] for i in ee_cols]

#     result_df = df[["t"] + ee_names].copy()
#     result_df.columns = ["t"] + [f"s{i+1}" for i in range(len(ee_cols))]

#     # compute velocities
#     print(result_df["t"].values)
#     t = result_df["t"].values.astype(float)
#     for col in result_df.columns[1:]:
#         result_df[f"{col}_dot"] = np.gradient(result_df[col].values.astype(float), t)

#     return result_df
def load_and_compute_velocities(csv_path: str, ee_cols=(8, 9, 10)):
    df = pd.read_csv(csv_path, header=None, skiprows=1)  # skip header if present

    num_cols = df.shape[1]
    df.columns = ["t"] + [f"c{i}" for i in range(1, num_cols)]

    ee_names = [df.columns[i] for i in ee_cols]
    result_df = df[["t"] + ee_names].copy()
    result_df.columns = ["t"] + [f"s{i+1}" for i in range(len(ee_cols))]

    # ensure numeric conversion
    result_df["t"] = pd.to_numeric(result_df["t"], errors="coerce")
    result_df.dropna(subset=["t"], inplace=True)

    t = result_df["t"].values.astype(float)
    for col in result_df.columns[1:]:
        result_df[f"{col}_dot"] = np.gradient(result_df[col].astype(float), t)

    return result_df



def generate_diverse_trajectories(df: pd.DataFrame, H: int, pos_range: float = 0.01, noise_std: float = 1e-3):
    """
    Generate H synthetic trajectories (positions+velocities) from a single trajectory df.
    Each trajectory gets a random mean shift (uniform in [-pos_range, pos_range]) plus small noise.
    Returns concatenated DataFrame of shape (H*T, columns).
    """
    df_all_list = []
    t = df["t"].values.astype(float)
    # determine number of position states from df (columns include velocities too)
    # If df has s1, ..., sN and s1_dot... then the number of unique position columns = (df.shape[1]-1)/2
    # but if df currently only has positions, it will have no _dot columns; handle both cases.
    # We expect input df to have only positions (no _dot), as produced by original CSV.
    # If velocities present, we recompute them after shifting.
    # Count position columns as those without "_dot"
    pos_cols = [c for c in df.columns if c != "t" and not c.endswith("_dot")]
    n_states = len(pos_cols)
    for h in range(H):
        df_new = df.copy()
        # Random mean shift for each position state
        mean_shifts = np.random.uniform(-pos_range, pos_range, size=n_states)
        for i in range(n_states):
            col = pos_cols[i]
            df_new[col] = df_new[col] + mean_shifts[i] + np.random.normal(0, noise_std, size=len(df))
        # Recompute velocities from the perturbed positions
        for i in range(n_states):
            col = pos_cols[i]
            df_new[f"{col}_dot"] = np.gradient(df_new[col].values.astype(float), t)
        df_all_list.append(df_new)
    df_all = pd.concat(df_all_list, ignore_index=True)
    return df_all

def generate_diverse_trajectories_v2(df: pd.DataFrame, H: int, pos_range: float = 0.01, noise_std: float = 1e-3, time_scale_range=(0.9, 1.1)):
    df_all_list = []
    t_original = df["t"].values.astype(float)
    pos_cols = [c for c in df.columns if c != "t" and not c.endswith("_dot")]
    n_states = len(pos_cols)

    for h in range(H):
        # 1) Time scaling factor
        scale = np.random.uniform(*time_scale_range)
        t_scaled = t_original * scale

        df_new = df.copy()
        df_new["t"] = t_scaled

        # 2) Random mean shift and additive noise
        mean_shifts = np.random.uniform(-pos_range, pos_range, size=n_states)
        for i, col in enumerate(pos_cols):
            df_new[col] = df_new[col] + mean_shifts[i] + np.random.normal(0, noise_std, size=len(df))

        # 3) Recompute velocities after position perturbation
        for col in pos_cols:
            df_new[f"{col}_dot"] = np.gradient(df_new[col].values.astype(float), t_scaled)

        df_all_list.append(df_new)

    df_all = pd.concat(df_all_list, ignore_index=True)
    return df_all

def plot_diverse_trajectories_time_scaled(df_diverse, H, n_states=3):
    """
    Plot H trajectories with possible time scaling (speed variations).
    """
    fig, axs = plt.subplots(n_states, 1, figsize=(10, 3*n_states), sharex=True)

    traj_len = df_diverse.shape[0] // H  # approximate length per trajectory
    for i in range(n_states):
        for h in range(H):
            df_traj = df_diverse.iloc[h*traj_len:(h+1)*traj_len]
            axs[i].plot(df_traj["t"].values, df_traj[f"s{i+1}"].values, alpha=0.7)
        axs[i].set_ylabel(f"s{i+1}")
        axs[i].grid(True)
    axs[-1].set_xlabel("time [s]")
    plt.tight_layout()
    # plt.show()



def prepare_data(df_diverse: pd.DataFrame, H: int, T: int, n_states: int):
    """
    Prepare MATLAB-like Data matrix and sklearn-ready data for GMM.
    Inputs:
      df_diverse: concatenated DataFrame with columns ['t', s1..sN, s1_dot..sN_dot]
                 arranged as H blocks each of length T
      H: number of trajectories
      T: length of each trajectory
      n_states: number of position states (N)
    Returns:
      Data_mat: numpy array shaped (D, H*T) where D = 1 + 2*n_states (MATLAB-like)
      data_sklearn: numpy array shaped (H*T, D) (samples x features) for sklearn
    """
    # Validate expected columns
    pos_cols = [f"s{i+1}" for i in range(n_states)]
    vel_cols = [f"s{i+1}_dot" for i in range(n_states)]
    required = ["t"] + pos_cols + vel_cols
    for c in required:
        if c not in df_diverse.columns:
            raise ValueError(f"Column {c} not found in df_diverse. Dataframe columns: {list(df_diverse.columns)}")

    # Extract in the MATLAB ordering: time first, then positions, then velocities
    data_arr = df_diverse[["t"] + pos_cols + vel_cols].values  # shape (H*T, D)
    Data_mat = data_arr.T  # shape (D, H*T)
    data_sklearn = data_arr.copy()  # (n_samples, n_features) as required by sklearn
    return Data_mat, data_sklearn


def fit_gmm(data, n_components=8, covariance_type='full', random_state=0, n_init=5, max_iter=200):
    """
    Fit a GaussianMixture (sklearn) on data (n_samples, n_features).
    Returns the fitted GaussianMixture object.
    """
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type=covariance_type,
                          random_state=random_state,
                          n_init=n_init,
                          max_iter=max_iter)
    gmm.fit(data)
    return gmm


def gmr(gmm: GaussianMixture, X_query: np.ndarray, in_idx=[0], out_idx=None, eps=1e-8):
    """
    Gaussian Mixture Regression (GMR).
    Parameters
    ----------
    gmm : sklearn.mixture.GaussianMixture (fitted)
    X_query : (M, d_in) array of query inputs (each row a d_in-dim input)
    in_idx : list or array of indices (indices of input dims in the original GMM features)
             Default [0] meaning first column is the input (time)
    out_idx : list or array of indices (indices of output dims)
              If None, out_idx = all indices not in in_idx.
    eps : small regularizer for matrix inversions
    Returns
    -------
    Mu : (M, D_out) predicted conditional mean for each query
    Sigma : (M, D_out, D_out) predicted conditional covariance for each query
    """
    # Validate and normalize indices
    D = gmm.means_.shape[1]
    in_idx = np.array(in_idx, dtype=int)
    if out_idx is None:
        out_idx = np.array([i for i in range(D) if i not in in_idx], dtype=int)
    else:
        out_idx = np.array(out_idx, dtype=int)

    M = X_query.shape[0]
    D_out = len(out_idx)

    # Pre-extract GMM parameters
    K = gmm.weights_.shape[0]
    weights = gmm.weights_           # shape (K,)
    means = gmm.means_               # shape (K, D)
    covs = gmm.covariances_          # shape (K, D, D) if covariance_type='full'
    # import pdb; pdb.set_trace()
    # Storage for outputs
    Mu = np.zeros((M, D_out))
    Sigma = np.zeros((M, D_out, D_out))

    # For numerical stability, ensure covs are symmetric
    for k in range(K):
        covs[k] = 0.5 * (covs[k] + covs[k].T)

    # Loop over query points
    for m in range(M):
        x = np.atleast_1d(X_query[m]).reshape(-1)  # shape (d_in,)
        # Compute responsibilities p(x | k) * pi_k
        pks = np.zeros(K)
        cond_means = np.zeros((K, D_out))
        cond_covs = np.zeros((K, D_out, D_out))
        for k in range(K):
            mu_k = means[k]           # shape (D,)
            Sigma_k = covs[k]        # shape (D, D)
            # Partition mu and Sigma
            mu_I = mu_k[in_idx]
            mu_O = mu_k[out_idx]
            Sigma_II = Sigma_k[np.ix_(in_idx, in_idx)]
            Sigma_IO = Sigma_k[np.ix_(in_idx, out_idx)]
            Sigma_OI = Sigma_k[np.ix_(out_idx, in_idx)]
            Sigma_OO = Sigma_k[np.ix_(out_idx, out_idx)]

            # Invert Sigma_II (regularize)
            try:
                Sigma_II_reg = Sigma_II + eps * np.eye(Sigma_II.shape[0])
                inv_Sigma_II = np.linalg.inv(Sigma_II_reg)
            except np.linalg.LinAlgError:
                inv_Sigma_II = np.linalg.pinv(Sigma_II + eps * np.eye(Sigma_II.shape[0]))

            # conditional mean and covariance for component k
            mu_k_cond = mu_O + Sigma_OI @ (inv_Sigma_II @ (x - mu_I))
            Sigma_k_cond = Sigma_OO - Sigma_OI @ (inv_Sigma_II @ Sigma_IO)

            cond_means[k, :] = mu_k_cond
            # enforce symmetry + small regularization
            Sigma_k_cond = 0.5 * (Sigma_k_cond + Sigma_k_cond.T) + eps * np.eye(D_out)
            cond_covs[k, :, :] = Sigma_k_cond

            # compute p(x | k) using multivariate normal on input part
            try:
                pks[k] = weights[k] * multivariate_normal.pdf(x, mean=mu_I, cov=Sigma_II_reg)
            except Exception:
                # fallback to small value if pdf fails
                pks[k] = weights[k] * 1e-300

        # normalize responsibilities h_k(x)
        denom = np.sum(pks)
        if denom <= 0:
            # numerical degenerate case: fallback to equal responsibilities
            h = np.ones(K) / K
        else:
            h = pks / denom  # shape (K,)

        # mixture mean: weighted sum of component conditional means
        mu_x = np.sum(h[:, None] * cond_means, axis=0)  # (D_out,)
        # mixture covariance: sum_k h_k [Sigma_k_cond + mu_k_cond mu_k_cond^T] - mu_x mu_x^T
        Sigma_x = np.zeros((D_out, D_out))
        for k in range(K):
            mm = cond_means[k][:, None]  # (D_out,1)
            Sigma_x += h[k] * (cond_covs[k] + mm @ mm.T)
        Sigma_x = Sigma_x - np.outer(mu_x, mu_x)
        # symmetrize and regularize
        Sigma_x = 0.5 * (Sigma_x + Sigma_x.T) + eps * np.eye(D_out)

        Mu[m] = mu_x
        Sigma[m] = Sigma_x

    return Mu, Sigma


def compute_reference_trajectory(gmm: GaussianMixture, time_query: np.ndarray, in_idx=[0], out_idx=None):
    """
    Query GMR on arbitrary time points to build the reference trajectory.

    Args:
        gmm: fitted GaussianMixture model
        time_query: (M, 1) array of time points at which to query GMR
        in_idx: list of input indices in GMM (default: [0] for time)
        out_idx: list of output indices in GMM (default: None -> all except in_idx)

    Returns:
        Mu_ref: (M, D_out) means at each query time
        Sigma_ref: (M, D_out, D_out) covariances at each query time
    """
    Mu_ref, Sigma_ref = gmr(gmm, time_query, in_idx=in_idx, out_idx=out_idx)
    return Mu_ref, Sigma_ref



# -------------------------
# Example usage (if run directly)
# -------------------------
if __name__ == "__main__":
    # ---- user parameters ----
    csv_path = "/home/iisc-hiro-lab-phd-2/Documents/KMP/scripts/paper_roboface_demonstrated_joint_positions(t,j1-j7,x,y,z).csv"  # change to your file
    H = 20                 # number of synthetic demos to generate
    pos_range = 0.02       # random mean shift range (meters)
    noise_std = 2e-3       # additive Gaussian noise std
    n_states = 3           # number of position dims in the demo
    nbStates = 10           # number of GMM components
    # -------------------------

    # 1) load single demo and compute velocities
    df = load_and_compute_velocities(csv_path, ee_cols=(8, 9, 10))  
    T = df.shape[0]
    demo_dt = np.median(np.diff(df["t"].values))
    print("Loaded demo, timesteps:", T, "dt:", demo_dt)

    # 2) generate diverse synthetic trajectories
    df_diverse = generate_diverse_trajectories_v2(df, H=H, pos_range=pos_range, noise_std=noise_std)
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
    # nbData = T
    # dt = demo_dt
    # Mu_ref, Sigma_ref = compute_reference_trajectory(gmm, nbData=nbData, dt=dt, in_idx=[0], out_idx=None)
    # Instead of nbData, dt:
    t_query = np.linspace(df_diverse["t"].min(), df_diverse["t"].max(), 2*len(df))  # for smooth output
    Mu_ref, Sigma_ref = compute_reference_trajectory(gmm, time_query=t_query, in_idx=[0], out_idx=None)

    print("Mu_ref shape:", Mu_ref.shape)        # (T, D_out)
    print("Sigma_ref shape:", Sigma_ref.shape)  # (T, D_out, D_out)

    # 6) quick sanity plot: first position dimension mean +- 2*sigma
    t = np.linspace(df_diverse["t"].min(), df_diverse["t"].max(), 2*len(df)) 
    # output dims: D_out = 2*n_states -> first n_states are positions, next are velocities
    D_out = Mu_ref.shape[1]
    assert D_out == 2 * n_states, "Expected output dim 2*n_states"

    plt.figure(figsize=(8, 4))
    idx = 2  # plot first position dim
    pos_mean = Mu_ref[:, idx]
    pos_std = np.sqrt(Sigma_ref[:, idx, idx])
    plt.plot(t, pos_mean, 'r-', label="GMR mean (pos1)")
    plt.fill_between(t, pos_mean - 2*pos_std, pos_mean + 2*pos_std, color='r', alpha=0.25, label="±2σ")
    # Overlay a few of the synthesized demonstrations (first trajectory only)
    traj_len = T
    for h in range(min(5, H)):
        start = h*traj_len
        end = (h+1)*traj_len
        plt.plot(df_diverse["t"].values[start:end], df_diverse[f"s{idx+1}"].values[start:end], alpha=0.5, linestyle='--')
    plt.xlabel("time [s]")
    plt.ylabel("position")
    plt.legend()
    plt.title("GMR reference mean ± 2σ (pos dimension 1) and a few demo samples")
    plt.tight_layout()
    plt.show()

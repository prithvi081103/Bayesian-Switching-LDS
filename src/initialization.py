"""
Initialization pipeline for RecurrentSLDS following Linderman et al. (2017) supplement §2:
probabilistic PCA / factor-style continuous states, AR-HMM on latent trajectory for z,
then logistic regression on x_t -> z_{t+1} to seed stick-breaking recurrence parameters.
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .switching_ar import BayesianSwitchingAR1

def initialize_rslds(
    y: ArrayLike,
    K: int,
    state_dim: int,
    ar_order: int = 1,
    *,
    n_gibbs_ar: int = 30,
    random_state: int | None = None,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize continuous states, discrete states, and recurrence (W, r) for RecurrentSLDS.
    """
    if K < 1:
        raise ValueError("K must be >= 1.")
    if state_dim < 1:
        raise ValueError("state_dim must be >= 1.")
    if ar_order < 1:
        raise ValueError("ar_order must be >= 1.")

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    t_y, d_obs = y.shape
    if t_y < ar_order + 2:
        raise ValueError("Time series too short for PCA + AR-HMM initialization.")

    # Step 1: PCA to reduce dimensions
    if standardize:
        y_in = StandardScaler().fit_transform(y)
    else:
        y_in = y - np.mean(y, axis=0, keepdims=True)

    pca = PCA(n_components=state_dim, random_state=random_state)
    x_init = pca.fit_transform(y_in).astype(float)
    t_x, m = x_init.shape
    assert m == state_dim

    # Step 2: Build trajectory for the AR-HMM
    if ar_order == 1:
        y_ar = x_init
        offset = 0
    else:
        d_ar = state_dim * ar_order
        t_ar = t_x - ar_order + 1
        y_ar = np.zeros((t_ar, d_ar))
        for i in range(t_ar):
            t0 = i + ar_order - 1
            blocks = [x_init[t0 - lag] for lag in range(ar_order)]
            y_ar[i] = np.concatenate(blocks, axis=0)
        offset = ar_order - 1

        # Step 3: Run the Bayesian AR-HMM on the PCA trajectory
    rng_state = np.random.get_state()
    try:
        if random_state is not None:
            np.random.seed(int(random_state))
        ar_model = BayesianSwitchingAR1(K=K, D=y_ar.shape[1])
        hist_ar = ar_model.gibbs(y_ar, n_iters=n_gibbs_ar)
    finally:
        np.random.set_state(rng_state)

    z_short = np.asarray(hist_ar["last_z"], dtype=int)

    z_init = np.zeros(t_x, dtype=int)
    z_init[:offset] = z_short[0] if z_short.size else 0
    z_init[offset : offset + len(z_short)] = z_short
    if offset + len(z_short) < t_x:
        z_init[offset + len(z_short) :] = z_short[-1]

    # Step 4: Logistic Regression for Recurrent parameters (W, r)
    k1 = max(0, K - 1)
    W = np.zeros((K, k1, state_dim), dtype=float)
    r = np.zeros((K, k1), dtype=float)

    if k1 == 0:
        return x_init, z_init, W, r

    X_pairs = x_init[:-1]
    z_next = z_init[1:]
    n_pairs = X_pairs.shape[0]
    if n_pairs < 2:
        return x_init, z_init, W, r

    for k in range(K):
        y_bin = (z_next == k).astype(int)
        if np.unique(y_bin).size < 2:
            coef = np.zeros(state_dim, dtype=float)
            intercept = 0.0
        else:
            clf = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=500,
                random_state=random_state,
            )
            clf.fit(X_pairs, y_bin)
            coef = np.asarray(clf.coef_, dtype=float).ravel()
            intercept = float(clf.intercept_[0])

        if k < k1:
            W[:, k, :] = coef
            r[:, k] = intercept

    return x_init, z_init, W, r


def apply_rslds_initialization(
    model,
    W: np.ndarray,
    r: np.ndarray,
) -> None:
    """Copy initialized ``W`` and ``r`` into ``model`` (in-place)."""
    model.W[:] = W
    model.r[:] = r

import math
from joblib import Parallel, delayed

import tensorflow as tf
import numpy as np

from multisol.tf.metrics import (
    accuracy,
    precision,
    recall,
    specificity,
    f1_score_fun,
    tss,
    gmean,
)


# ============================================================================
# Differentiable Monte Carlo indicator for multi-class assignment
#
# For a given sample with prediction vector y_pred (in S_m) and tau samples from
# a Dirichlet on the simplex, we approximate:
#
#    φ_j(y_pred, τ) = ∏_{k≠j} σ(λ [ y_pred^j - y_pred^k - (τ^j - τ^k) ])
#
# Averaging over N tau samples gives:
#
#    ψ_j(y_pred) = (1/N) ∑_{l=1}^{N} φ_j(y_pred, τ^(l)).
#
# This ψ_j is used in a one-vs-rest confusion matrix.
# ============================================================================
@tf.function
def multiclass_indicator(y_pred, taus, lam=10.0):
    """
    y_pred:  Tensor of shape (B, m) -- softmax outputs.
    taus:    Tensor of shape (N, m) -- tau samples from a distribution on S_m.
    lam:     Sigmoid steepness parameter.

    Returns:
       psi: Tensor of shape (B, m) with the differentiable assignment probabilities.
    """
    # Use static shape if available.
    m_static = y_pred.shape[1]
    m_static if m_static is not None else tf.shape(y_pred)[1]

    # Compute pairwise differences for predictions: shape (B, m, m).
    diff_y = y_pred[:, :, None] - y_pred[:, None, :]

    # Compute pairwise differences for taus: shape (N, m, m).
    diff_tau = taus[:, :, None] - taus[:, None, :]

    # Combine differences: shape (B, N, m, m).
    diff = diff_y[:, None, :, :] - diff_tau[None, :, :, :]

    # Apply sigmoid.
    s = tf.sigmoid(lam * diff)  # shape (B, N, m, m)

    # Set diagonal (j==k) to 1
    s = tf.linalg.set_diag(s, tf.ones(tf.shape(s)[:-1], dtype=s.dtype))

    # Product over k (axis=-1) and average over tau samples.
    prod = tf.reduce_prod(s, axis=-1)  # shape (B, N, m)
    psi = tf.reduce_mean(prod, axis=1)  # shape (B, m)
    return psi


# ============================================================================
# SOL Loss Function for Multi-class (one-vs-rest) case
# ============================================================================
tf.config.optimizer.set_jit(True)


def count_condition_satisfied(x, i, taus):
    """
    Count how many Dirichlet samples satisfy the condition:
    x_i - x_j > tau^k_i - tau^k_j for all j ≠ i, using optimization.

    Parameters:
        x (list): Point in the simplex.
        i (int): Fixed component.
        taus (ndarray): Dirichlet samples, shape (N, d).

    Returns:
        int: Count of samples satisfying the condition.
    """
    N, d = taus.shape

    # Compute differences: x_i - x_j for all j ≠ i
    x_diff = x[i] - x  # Shape (d,)

    def check_condition_for_tau(tau):
        # Check the condition for all j ≠ i for a single tau sample
        for j in range(d):
            if j == i:
                continue
            if not (x_diff[j] > (tau[i] - tau[j])):
                return False
        return True

    # Use parallel processing to evaluate the condition for all tau samples
    condition_satisfied = Parallel(n_jobs=-1)(delayed(check_condition_for_tau)(tau) for tau in taus)

    # Return the count of samples that satisfy the condition
    return np.sum(condition_satisfied)

def SOL(
    score="accuracy",
    taus=None,
    lam=10.0,
):
    """
    Score-Oriented Loss (SOL) for multi-class classification.

    Computes a soft one-vs-rest confusion matrix by approximating the indicator
    1{ y_pred in R_j(τ) } via Monte Carlo integration over tau samples.

    Parameters:
      score:        String indicating the score (e.g., 'accuracy').
      mu, delta:    Parameters for the cosine distribution (ignored if 'uniform').
      mode:         'average' for macro averaging.
      taus:         A NumPy array or tensor of shape (N, m) containing tau samples.
      lam:          Sigmoid steepness parameter.

    Returns:
      A loss function SOL_(y_true, y_pred) for model.compile.
    """
    # Select score function from metrics.
    if score == "accuracy":
        score_func = accuracy
    elif score == "precision":
        score_func = precision
    elif score == "recall":
        score_func = recall
    elif score == "specificity":
        score_func = specificity
    elif score == "f1_score":
        score_func = f1_score_fun
    elif score == "tss":
        score_func = tss
    elif score == "gmean":
        score_func = gmean
    else:
        score_func = accuracy  # default

    taus = tf.convert_to_tensor(taus, dtype=tf.float32)

    @tf.function
    def SOL_(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Binary classification branch.
        if y_pred.shape[1] == 1:
            TN = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))
            TP = tf.reduce_sum(y_true * y_pred)
            FP = tf.reduce_sum((1.0 - y_true) * y_pred)
            FN = tf.reduce_sum(y_true * (1.0 - y_pred))
            return -score_func(TN, FP, FN, TP) + 1.0
        else:
            # Multi-class branch.
            psi = multiclass_indicator(y_pred, taus, lam=lam)  # shape (B, m)

            # Compute confusion matrix components for each class.
            TP = tf.reduce_sum(y_true * psi, axis=0)
            FN = tf.reduce_sum(y_true * (1.0 - psi), axis=0)
            FP = tf.reduce_sum((1.0 - y_true) * psi, axis=0)
            TN = tf.reduce_sum((1.0 - y_true) * (1.0 - psi), axis=0)

            # Compute the score per class and average.
            score_arr = score_func(TN, FP, FN, TP)
            final_score = tf.reduce_mean(score_arr)
            return -final_score + 1.0

    return SOL_

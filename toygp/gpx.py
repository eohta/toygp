
#
# Toy GP - Extra Functions
#

import numpy as np

import jax
import jax.numpy as jnp

import numpyro

from toygp import gp

def sparse_fitc(kf, x, xu, noise_sd, jitter=1.0e-6):

    sigma2 = jnp.square(noise_sd)

    W = gp.cov_factor_sparse(kf, x, xu, jitter)
    D = gp.cov_diag_sparse(kf, x, W)

    D = D + sigma2 * jnp.ones(x.shape[0])

    return W, D

def sparse_vfe(kf, x, xu, noise_sd, jitter=1.0e-6):

    sigma2 = jnp.square(noise_sd)

    W = gp.cov_factor_sparse(kf, x, xu, jitter)
    C = gp.cov_diag_sparse(kf, x, W)

    trace_term = -0.5 * C.sum() / sigma2
    numpyro.factor('trace_term', trace_term)

    D = jnp.ones(x.shape[0]) * sigma2

    return W, D

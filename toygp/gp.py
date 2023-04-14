#
# ToyGP - Base Functions
#

import numpy as np

import jax
import jax.numpy as jnp

import numpyro

from scipy.cluster.vq import kmeans2

#
# jax enable x64
#

jax.config.update('jax_enable_x64', True)

#
# Cov factor(W) and diag(D) 
#

def cov_factor_exact(kf, x, jitter=1.0e-6):

    K = kf(x, x)
    K = K + jitter * jnp.eye(K.shape[0])
    W = jnp.linalg.cholesky(K)

    return W

def cov_factor_sparse(kf, x, xu, jitter=1.0e-6):

    Kuu = kf(xu, xu)
    Kuu = Kuu + jitter * jnp.eye(Kuu.shape[0])
    Luu = jnp.linalg.cholesky(Kuu)

    Kuf = kf(xu, x)
    W = jax.scipy.linalg.solve_triangular(Luu, Kuf, lower=True).transpose()

    return W

def cov_diag_sparse(kf, x, W):

    diag_Kff = jax.vmap(kf, (0, 0), 0)(x, x)
    diag_Kff = diag_Kff.flatten()

    diag_Qff = jnp.square(W).sum(axis=1)
    diag_Qff = diag_Qff.flatten()

    D = diag_Kff - diag_Qff
    D = jnp.clip(D, 0.0, jnp.inf)

    return D

#
# Noise-excluded form of GP
#

def exact(kf, x, v, jitter=1.0e-6):

    W = cov_factor_exact(kf, x, jitter)
    f = v @ W.transpose()

    return f

def sparse(kf, x, xu, v, jitter=1.0e-6):

    W = cov_factor_sparse(kf, x, xu, jitter)
    f = v @ W.transpose()

    return f

#
# Noise-included form of GP
#

def sparse_fitc(kf, x, xu, v, noise_sd=0, jitter=1.0e-6):

    W = cov_factor_sparse(kf, x, xu, jitter)
    D = cov_diag_sparse(kf, x, W)

    f_mu = v @ W.transpose()
    f_sd = jnp.sqrt(D) + noise_sd

    return f_mu, f_sd

def sparse_vfe(kf, x, xu, v, noise_sd, jitter=1.0e-6):

    sigma2 = jnp.square(noise_sd)

    W = cov_factor_sparse(kf, x, xu, jitter)
    D = cov_diag_sparse(kf, x, W)

    trace_term = -0.5 * D.sum() / sigma2
    numpyro.factor('trace_term', trace_term)

    f_mu = v @ W.transpose()
    f_sd = jnp.ones(x.shape[0]) * noise_sd

    return f_mu, f_sd

#
# Inducing Points
#

def setup_inducing_kmeans(x, num_inducing_pts):

    xu = kmeans2(np.array(x), num_inducing_pts, minit="points")[0]
    xu = jnp.array(xu)

    return xu

def setup_inducing_subsample(key, x, num_inducing_pts):

    xu = jax.random.permutation(key, x)[:num_inducing_pts]

    return xu

def setup_inducing_normal(key, num_latent, num_inducing_pts):

    xu = jax.random.normal(key, shape=(num_inducing_pts, num_latent))

    return xu

#
# Kronecker Product
#

def col_slice_kron(A, B, k):

    [q, r] = jnp.divmod(k, B.shape[1])

    Aq = jax.lax.dynamic_slice(A, (0, q), (A.shape[0], 1))
    Br = jax.lax.dynamic_slice(B, (0, r), (B.shape[0], 1))

    return jnp.kron(Aq, Br)

def row_slice_kron(A, B, k):

    [q, r] = jnp.divmod(k, B.shape[0])

    Aq = A[q]
    Br = B[r]

    return jnp.kron(Aq, Br)

def kron_prod(Wx, Wy, v):

    Wx = Wx.transpose()
    Wy = Wy.transpose()

    N = Wx.shape[1] * Wy.shape[1]

    def body_func(carry, k):

        f_ = jnp.dot(v, col_slice_kron(Wy, Wx, k))[0]
        
        return carry, f_

    _, f = jax.lax.scan(body_func, None, jnp.arange(N))

    return f

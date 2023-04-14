#
# ToyGP - Kernel Functions
#

import jax.numpy as jnp

#
# Distances
#

def distance_L1(x, y, r=1.0):

    xx = jnp.atleast_2d(x) / r
    yy = jnp.atleast_2d(y) / r

    xx = jnp.expand_dims(xx, 1)
    yy = jnp.expand_dims(yy, 0)

    d = jnp.abs(xx - yy)
    d = jnp.atleast_3d(d)

    d = jnp.sum(d, axis=2)
    
    return d

def distance_L2(x, y, r=1.0):

    xx = jnp.atleast_2d(x) / r
    yy = jnp.atleast_2d(y) / r

    xx = jnp.expand_dims(xx, 1)
    yy = jnp.expand_dims(yy, 0)

    d = jnp.square(xx - yy)
    d = jnp.atleast_3d(d)

    d = jnp.sum(d, axis=2)
    
    return d

#
# Kernels
#

def constant(x, y):

    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)

    N = x.shape[0]
    M = y.shape[0]
    K = jnp.ones([N, M])

    return K

def linear(x, y):

    K = jnp.inner(x, y)

    return K

def exponential(x, y, r):

    D = distance_L1(x, y, r)
    K = jnp.exp(-D)

    return K

def rbf(x, y, r):

    D = distance_L2(x, y, r)
    K = jnp.exp(-0.5 * D)

    return K

def wgn(x, y):

    D = distance_L1(x, y, 1)
    K = (D < jnp.finfo(jnp.float64).eps).astype(jnp.float64)

    return K

def matern32(x, y, r):

    c = jnp.sqrt(3)

    D = distance_L1(x, y, r)
    K = (1.0 + c) * jnp.exp(-c * D)

    return K

def matern52(x, y, r):

    c1 = jnp.sqrt(5.0)
    c2 = 5.0 / 3.0

    D1 = distance_L1(x, y, r)
    D2 = jnp.square(D1)

    K = (1.0 + c1 * D1 + c2 * D2) * jnp.exp(-c1 * D1)

    return K

def cosine(x, y, r):

    D = distance_L1(x, y, r)
    K = jnp.cos(2.0 * jnp.pi * D)

    return K

def exp_cosine(x, y, r):

    D = distance_L1(x, y, r)
    K = jnp.exp(jnp.cos(2.0 * jnp.pi * D))

    return K

def rat_quad(x, y, r, alpha):

    D = distance_L2(x, y, r)
    K = jnp.power(1 + (0.5 / alpha) * D, -alpha)

    return K

def periodic(x, y, period, r):

    D = distance_L1(x, y, period)
    E = -2.0 * jnp.square(jnp.sin(jnp.pi * D) / r)
    K = jnp.exp(E)

    return K

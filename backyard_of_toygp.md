# Backyard of ToyGP

Here, I'm going to explain what I thought behind this package...

## About Categories of GPs

Typically GP fall into these two categories:

- exact GP
- sparse GP

and most of Python packages have their interface along this categorization. For example, Pyro follows this categorization and supports these 3 methods for sparse GP.

- DTC
- FITC
- VFE

If you look at Pyro's source code, you will find that Pyro implements them in a function quite smartly using the LowRankMultivariteNormal distribution.

https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/gp/models/sgpr.py 

```
# W = (inv(Luu) @ Kuf).T 
# Qff = Kfu @ inv(Kuu) @ Kuf = W @ W.T 

# Fomulas for each approximation method are 
# DTC:  y_cov = Qff + noise,                   trace_term = 0 
# FITC: y_cov = Qff + diag(Kff - Qff) + noise, trace_term = 0 
# VFE:  y_cov = Qff + noise,                   trace_term = tr(Kff-Qff) / noise

# y_cov = W @ W.T + D
# trace_term is added into log_prob
```

Personally I really like this implementation, and this is one of the most awesome comments I've ever seen. But after a lot of thinking, I decide to implement GP in a slightly different way. I decided to implement GP along this categorization.

### Noise-excluded form of GP
- exact GP
- sparse GP (DTC w/o noise)
 
### Noise-included form of GP
- sparse GP (FITC)
- sparse GP (VFE)

This may seem a little strange, but for me there is a good reason. Let me explain.

## Noise-excluded form of Sparse GP (DTC w/o noise)

According to Joaquin Quinonero-Candela and Carl E. Rasmussen [2], the DTC approximation of a sparse GP can be understood as a generative model:

$$\tag{1} u \sim N(0, K_{uu})$$

$$\tag{2} f | u \sim N(K_{uf}^{T} K_{uu}^{-1} u, 0)$$

$$\tag{3} y | f \sim N(f, \sigma^{2} I)$$

And this generative model would also be expressed like this

$$\tag{4} v \sim N(0, I)$$

$$\tag{5} f = W v$$

$$\tag{6} y | f \sim N(f, \sigma^{2} I)$$

if you set $u = L_{uu}v, W = (L_{uu}^{-1} K_{uf})^{T}$.

You can see that this generative process becomes quite simple & straightforward like exact GP. And I thought that exact GP and sparse GP could share a similar user interface, based on this consideration.

And if you think of (6) as an "observer model", you can replace this Gaussian distribution with any kind of distribution, such as Poisson or Student's t-distribution.

Personally, this kind of GP (Noise excluded form of GP) can be used for various kind of GP application including latent variable model without a lot of thinking.

## Noise-included form of Sparse GP (FITC)

If you look at FITC as a generative model, you can see that it inevitably contains a noise term.

$$\tag{7} u \sim N(0, K_{uu})$$

$$\tag{8} f | u \sim N(K_{uf}^{T} K_{uu}^{-1} u, diag(K_{ff} - Q_{ff}))$$

$$\tag{9} y | f \sim N(f, \sigma^{2} I)$$

Unlike eq.(6), eq.(8) cannot be expressed in an exact form. By including $diag(K_{ff} - Q_{ff})$ in eq.(9), this generative model can also be expressed like this:

$$\tag{10} v \sim N(0, I)$$

$$\tag{11} f = W v$$

$$\tag{12} y | f \sim N(f, \sigma^{2} I + diag(K_{ff} - Q_{ff}))$$

But this can only be done if the observation of this model is Gaussian. And its implementation seemed to be quite straightforward like this:

```
v = numpyro.sample('v', dist.Normal(0, 1), sample_shape=(M,))
kf = lambda x0, x1 : kernels.rbf(x0, x1)
f_mu, f_sd = gp.sparse_fitc(kf, X, Xu, jitter)
f = numpyro.sample('f', dist.Normal(f_mu, f_sd).to_event(1))
```

But somehow this interface doesn't work well with GPLVM. So I also implemented another inferfaces like this:

```
v = numpyro.sample('v', dist.Normal(0, 1), sample_shape=(M,))
kf = lambda x0, x1 : kernels.rbf(x0, x1)
W, D = gpx.sparse_fitc(kf, X, Xu, jitter)
f = numpyro.sample('f', dist.LowRankMultivariateNormal(loc=jnp.zeros(N), cov_factor=W, cov_diag=D))
```
The implementation using the LowRankMultivariateNormal distribution is adapted in most Python packages, and it seems to be a fairly standard way to implement sparse GP. But unfortunately, you can't use LowRankMultivariateNormal for "Noise excluded form of GP".

## What is LowRankMultivariateNormal distribution ?

Typically, the sparse GP implementation uses a distribution called "LowRankMultivariateNormal". This is a MultivariateNormal distribution that has a different parameterization than the standard one. When the covariance matrix K is formed this way, 

$$K = W W^T + D$$

you can construct multivariate normal distribution using this LowRankMultivariateNormal class like this:

```
dist.LowRankMultivariateNormal(jnp.zeros(N), cov_factor=W, cov_diag=D)
```

But you can't define this distribution if cov_diag is a zero vector, which means that "Noise-excluded form of GP" cannot be implemented with this distribution. For this category of GP, personally I think using standard Normal distribution is the natural way to implement this GPs.

## Summary

From a modeling and implementation perspective, it seems more important whether the GP is defined with noise than whether it is sparse or not. And I categorized GPs into two categories.

- Noise-excluded form of GP
- Noise-included form of GP

And in ToyGP, GP functions are implemented into two different types.

- Type1 : Implementation based on standard normal distribution
- Type2 : Implementation based on LowRankMultivariateNormal distribution

Type1 interface supports both "Noise-excluded form of GP" and "Noise-included form of GP". But Type2 interface only supports "Noise-included form of GP".

| Type of interface | Noise-excluded form | Noise-included form |
|-----------|----------------|----------------|
| Type1 | gp.exact <br> gp.sparse (DTC w/o noise) | gp.sparse_fitc (FITC) <br> gp.sparse_vfe (VFE) |
| Type2 | Not supported | gpx.sparse_fitc (FITC) <br> gpx.sparse_vfe (VFE) |

If you are new to GP, please start with the "Noise excluded form of GP" category. As long as you play in this category, things are simple and straightforward. And if you think you need FITC or VFE, please step up to "Noise-included form of GP".

But this is just my personal opinion, and I'm not an expert of GP. If you have any advice for me, please let me know. It would also be very much appreciated...

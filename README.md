# ToyGP

## What is ToyGP?

This is a toy to play with GP (Gaussian Process) and [NumPyro], which consists of mainly these files.

- [kernels.py]
- [gp.py]
- [gpx.py]

If you are looking for a more reliable package for GP, you can find real Python packages like [GPy], [GPyTorch], [Pyro], ... and so on. But if you are familiar with NumPyro and are just looking for a little toy to play with GP, this is probably for you.

This is very simple and small source code that you can play with and easily understand how it works.

## Why this is a toy?

There are two reasons why I think this is a toy. The first reason is that this software was written by a person who is neither an expert in GP nor in statistics. And the second reason is that this software has given up on one of the great aspects of GP ... **prediction**.

For example, if you use GP for regression, GP can very quickly predict its output using an analytical formula. And I think that's one of the greatest aspects of GP. But I just gave it up for simplicity. Instead, I fully rely on numpyro's great prediction mechanism for the prediction part of GP. That's a second reason why I think this is a toy.

And if you are looking for more reliable Jax-based GP packages that include prediction aspects, please consider [tinygp], which can also be used with NumPyro. And if you are looking for some source code related to Jax and GP, you can also find a lot of gems in [Juan Emmanuel Johnson's repository].

## How to install

To install this package, please follow these steps.

Step1) Clone this repository

```
git clone https://github.com/eohta/toygp.git
```

Step2) Move to the directory
```
cd ./toygp
```
Step3) Install package
```
pip install .
or
pip install -e .
```

Or just copying these two files into your directory might be enough in many cases...

- [kernels.py]
- [gp.py]
- [gpx.py]

Not to mention that you will need [NumPyro]. And you may also need seaborn to run some examples. Currently, I have checked that toygp works with NumPyro version 0.11. But it might also work with a lower version of NumPyro.

## Tutorial / How to use

To incorporate GP into your model, the only thing you need to know is how to sample GP with NumPyro.

GP is basically just a sample of the multivariate normal distribution:

$$
f \sim N(0, K)
$$

For simplicity, I will assume that this multivariate normal distribution has a mean of zero. K is the covariance matrix, also called the "kernel matrix" or "Gram matrix" in the GP context. And you can use the standard normal distribution to sample from this distribution:

$$v \sim N(0, I)$$

$$f = W v$$

$v$ is a vector of standard normal distribution, $W$ is a matrix which statisfy this condition:

$$K = WW^T$$

For this type of decomposition, typically one can typically use the Cholesky decomposition. W is a lower triangular matrix.
And you can write this process down like this:

```
v = numpyro.sample('v', dist.Normal(0, 1), sample_shape=(N,))
K = kernels.rbf(X, X)
W = jax.linalg.cholesky(K)
f = W @ v
```
If X is an N-dimensional vector, K becomes an (N, N)-matrix. And often we have to add a very small number to the diagonal components to avoid numerical problems.

```
v = numpyro.sample('v', dist.Normal(0, 1), sample_shape=(N,))
K = kernels.rbf(X, X) + jitter * jnp.eye(N)
W = jax.linalg.cholesky(K)
f = W @ v
```

If you use a ToyGP function, you can rewrite the same process like this:

```
v = numpyro.sample('v', dist.Normal(0, 1), sample_shape=(N,))
kf = lambda x0, x1 : kernels.rbf(x0, x1)
W = gp.cov_factor_exact(kf, X, jitter)
f = W @ v
```

But you can make this a little simpler using ToyGP function.

```
v = numpyro.sample('v', dist.Normal(0, 1), sample_shape=(N,))
kf = lambda x0, x1 : kernels.rbf(x0, x1)
f = gp.exact(kf, X, jitter)
```

Some may feel that ToyGP doesn't make things any simpler, but it does. You can write sparse GP code pretty much the same way.

```
v = numpyro.sample('v', dist.Normal(0, 1), sample_shape=(M,))
kf = lambda x0, x1 : kernels.rbf(x0, x1)
f = gp.sparse(kf, X, Xu, jitter)
```

That's it!

## Speed up GP

Sparse GP is one of the ways to speed up GP. Without sparse GP, computation sometimes becomes heavy even if you are dealing with less than a thousand data. So sparse GP is essential.

And sparse GP is also essential for another reason. ToyGP needs sparse GP if you want to predict output for arbitrary data points using NumPyro's prediction mechanism. I won't explain the details of this limitation here, but this is another reason why I think this package is a toy... But if you like, please enjoy this software 

Sparse GP needs auxiliary information called "inducing points" to speed up GP. In the code above, this information is expressed as Xu. You need choose Xu properly to use sparse GP.

And to speed up GP, you can also use the technique called "Kronecker methods". This method is typically used when you observe data along a 2-dim or 3-dim grid. For example, you can write code for data on a 2-dim grid like this:

```
v = numpyro.sample('v', dist.Normal(0, 1), sample_shape=(M,))
kf = lambda x0, x1 : kernels.rbf(x0, x1)
Wx = gp.cov_factor_sparse(kf, X, Xu, jitter)
Wy = gp.cov_factor_sparse(kf, Y, Yu, jitter)
f = gp.kron_prod(Wx, Wy, v)
```

Currently ToyGP only supports data on 2-dim grid. If you want to use faster GP for even higher dimensional grids, please consider other real Python packages out there which are developed by real smart people. In these packages, you can use the really fast method called KISS-GP method for example...

But until then, you can play with this toy. I have prepared some notebooks that can be used as tutorials.

Step1) Check how you can sample GP

- [How to sample GP - type1]

Step2) Check how you can use GP for regression

- [GPR - 1dim / gaussian / exact / svi]
- [GPR - 1dim / gaussian / sparse / svi]
- [GPR - 1dim / gaussian / sparse / mcmc]

## Sparse GP (FITC and VFE)

In the above explanation, I was dealing with sparse GP called "DTC" for simplicity. But if you are familiar with sparse GP, you might be thinking, "What about FITC or VFE? Does toygp support them?" And the answer would be "yes".

But because of FITC and VFE, sparse GP gets a little complicated from a modeling & implementation point of view, I think. So please stay away from them if you are new to GP. And if you really need them, please step up to FITC or VFE.

If you are interested in what I thought behind ToyGP, please check here: [backyard-of-toygp]

## Examples (Notebooks)

#### How to sample GP

- [How to sample GP - type1]
- [How to sample GP - type2]

#### Kernels:

- [Kernel Showcase]

#### GPR:

- [GPR - 1dim / gaussian / exact / svi]
- [GPR - 1dim / gaussian / sparse / svi]
- [GPR - 1dim / gaussian / sparse / mcmc]
- [GPR - 1dim / poisson / exact / svi]
- [GPR - 2dim / poisson / sparse / svi]
- [GPR - 2dim / poisson / sparse / kronecker / svi]
- [GPR - 2dim / gaussian / exact / kronecker / svi]

#### GPLVM

- [GPLVM - sparse / svi / iris-dataset]
- [GPLVM - sparse / svi / oil-flow-dataset]
- [GPLVM - sparse / svi / qPCR-dataset]


## Feedbacks & Comments

I think ToyGP has a lot of bugs at the moment. If you find any bugs, it would be greatly appreciated if you let me know from [here].

Sending me a pull request might also be helpful, but I'm not very familiar with the open source development culture or the github system. So I'm not sure I could handle it properly (but I'll try...).

And if you have any advice for me, please let me know. It would also be very much appreciated.

## Known Issues

ToyGP is still a work in progress. There are some unresolved issues...

#### Parameter Estimation Issue (MCMC vs SVI)
Currently ToyGP cannot estimate kernel parameters accurately, especially if you use SVI. You can estimate them more accurately if you use MCMC.

- [GPR - 1dim / gaussian / sparse / svi]
- [GPR - 1dim / gaussian / sparse / mcmc]

I don't know why this is happening...

#### GPLVM Issue
GPLVM does not work well with the type1 interface (standard normal distribution based implementation).

## Reference (English Papers)

[1] Gaussian Processes for Machine Learning, Carl E. Rasmussen, and Christopher K. I. Williams

[2] A Unifying View of Sparse Approximate Gaussian Process Regression, Joaquin Quinonero-Candela, and Carl E. Rasmussen

[3] Bayesian Gaussian Process Latent Variable Model, Michalis K. Titsias, Neil D. Lawrence

## Reference (Japanese Books)

There are some Japanese textbooks that I have also referenced. If you read Japanese, please check these [Japanese textbooks].
I think Daichi Mochihashi and Shigeyuki Ooba's textbook ([4]) is one of the most comprehensive textbooks for GP I've ever read.

## Mesage

Finally, I'd like to thank all the open source developers and researchers, especially those who created Jax and NumPyro. I'm just a big fan of these packages. I hope someone might enjoy this little toy until real interfaces for GP are implemented in NumPyro...

[backyard-of-toygp]:https://github.com/eohta/toygp/tree/main/backyard_of_toygp.md
[Japanese textbooks]:https://github.com/eohta/toygp/tree/main/toygp/japanese_textbooks.md
[here]:https://github.com/eohta

[How to sample GP - type1]:https://github.com/eohta/toygp/tree/main/notebooks/01_gp_base/01_sample_gp_type1.ipynb
[How to sample GP - type2]:https://github.com/eohta/toygp/tree/main/notebooks/01_gp_base/02_sample_gp_type2.ipynb
[Kernel Showcase]:https://github.com/eohta/toygp/tree/main/notebooks/01_gp_base/03_kernel_showcase.ipynb

[GPR - 1dim / gaussian / exact / svi]:https://github.com/eohta/toygp/tree/main/notebooks/02_gp_1dim/01_gaussian_exact_svi.ipynb
[GPR - 1dim / gaussian / sparse / svi]:https://github.com/eohta/toygp/tree/main/notebooks/02_gp_1dim/02_gaussian_sparse_svi.ipynb
[GPR - 1dim / gaussian / sparse / mcmc]:https://github.com/eohta/toygp/tree/main/notebooks/02_gp_1dim/03_gpr_1dim_gaussian_sparse_mcmc.ipynb
[GPR - 1dim / gaussian / sparse-vfe / svi]:https://github.com/eohta/toygp/tree/main/notebooks/02_gp_1dim/04_gaussian_sparse_vfe_svi.ipynb
[GPR - 1dim / poisson / exact / svi]:https://github.com/eohta/toygp/tree/main/notebooks/02_gp_1dim/05_poisson_exact_svi.ipynb

[GPR - 2dim / poisson / sparse / svi]:https://github.com/eohta/toygp/tree/main/notebooks/03_gp_2dim/01_poisson_sparse_svi.ipynb
[GPR - 2dim / poisson / sparse / kronecker / svi]:https://github.com/eohta/toygp/tree/main/notebooks/03_gp_2dim/02_poisson_sparse_kronecker_svi.ipynb
[GPR - 2dim / gaussian / exact / kronecker / svi]:https://github.com/eohta/toygp/tree/main/notebooks/03_gp_2dim/03_gaussian_exact_kronecker_svi.ipynb

[GPLVM - sparse / svi / iris-dataset]:https://github.com/eohta/toygp/tree/main/notebooks/04_gplvm/01_sparse_svi_iris.ipynb
[GPLVM - sparse / svi / oil-flow-dataset]:https://github.com/eohta/toygp/tree/main/notebooks/04_gplvm/02_sparse_svi_oil_flow.ipynb
[GPLVM - sparse / svi / qPCR-dataset]:https://github.com/eohta/toygp/tree/main/notebooks/04_gplvm/03_sparse_svi_qPCR.ipynb

[gp.py]:https://github.com/eohta/toygp/tree/main/toygp/gp.py
[gpx.py]:https://github.com/eohta/toygp/tree/main/toygp/gpx.py
[kernels.py]:https://github.com/eohta/toygp/tree/main/toygp/kernels.py

[tinygp]:https://github.com/dfm/tinygp
[Juan Emmanuel Johnson's repository]:https://github.com/jejjohnson
[jaxkern]:https://github.com/IPL-UV/jaxkern
[NumPyro]:https://github.com/pyro-ppl/numpyro
[Pyro]:https://github.com/pyro-ppl/pyro
[GPy]:https://github.com/SheffieldML/GPy
[GPyTorch]:https://gpytorch.ai/

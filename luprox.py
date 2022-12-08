from typing import Callable

from jax import eval_shape, jacfwd, jit
from jax import numpy as jnp
from jax import random
from scipy.io import savemat


def linear_uncertainty(fun: Callable):
    """Decorator to transform a function to return mean and covariance of output,
    using linear uncertainty propagation.

    See Measurements.jl in https://mitmath.github.io/18337/lecture19/uncertainty_programming

    Uncertainty propagation is only done for the first argument of the function,
    which is assumed to be a 1D array. The signature of the output function is
    `fun_with_uncertainty(mean, covariance, *args, **kwargs)`. Internally, the
    input function `fun` is called as `fun(mean, *args, **kwargs)`.

    Args:
        fun (Callable): Function to be transformed

    Returns:
        Callable: Transformed function
    """

    def fun_with_uncertainty(mean, covariance, *args, **kwargs):
        mean = mean.real
        covariance = covariance.real

        out_shape = eval_shape(fun, mean, *args, **kwargs).shape

        def f(x):
            y = fun(x, *args, **kwargs)
            return jnp.ravel(y)

        # Getting output meand and covariance
        out_mean = f(mean)
        J = jacfwd(f)(mean)

        out_cov = (jnp.abs(J) ** 2) * jnp.diag(
            covariance
        )  # Assumes diagonal covariance matrix
        out_cov = jnp.sum(
            out_cov, axis=-1
        )  # Summing variances as parameters are independent
        del J
        out_cov = jnp.reshape(out_cov, out_shape)
        out_mean = jnp.reshape(out_mean, out_shape)

        return out_mean, out_cov

    return jit(fun_with_uncertainty)


def monte_carlo(fun: Callable, trials, verbose=False, save_samples=True):
    """Decorator to transform a function to return mean and covariance of output,
    using Monte Carlo sampling.

    Uncertainty propagation is only done for the first argument of the function,
    which is assumed to be a 1D array. The signature of the output function is
    `fun_with_uncertainty(mean, covariance, key, *args, **kwargs)`. Internally, the
    input function `fun` is called as `fun(mean, *args, **kwargs)`.
    The key is used to generate random numbers for Monte Carlo sampling.

    Args:
        fun (Callable): Function to be transformed
        trials (int): Number of Monte Carlo trials
        verbose (bool, optional): Print progress. Defaults to False.
        save_samples (bool, optional): Save samples in .mat files. Defaults to True.

    Returns:
        Callable: Transformed function
    """

    def sampling_function(mean, covariance, key, *args, **kwargs):
        keys = random.split(key, trials)

        # This is standard deviation (uncorrelated variables)
        L = jnp.diag(jnp.sqrt(covariance))  # Assumes diagonal covariance matrix

        # For correlated variables, use
        # jnp.linalg.cholesky(covariance)

        meanval = 0
        x_squared = 0

        def _sample(mean, L, key):
            noisy_x = mean + L * random.normal(
                key, mean.shape
            )  # Product because L is a vector

            # If L is a matrix
            # noisy_x = mean + jnp.dot(L, random.normal(key, mean.shape))

            return fun(noisy_x, *args, **kwargs)

        for i in range(trials):
            if verbose:
                print(f"mc trial {i}")

            sample = _sample(mean, L, keys[i])

            if save_samples:
                mdic = {
                    "pressure": sample,
                }
                savemat(f"mc_{i}.mat", mdic)

            meanval = meanval + sample / trials
            x_squared = x_squared + (sample**2) / trials
            del sample

        var = x_squared - meanval**2
        var = var * trials / (trials - 1)
        return meanval, var

    return sampling_function


def mc_uncertainty(fun: Callable, trials):
    def fun_with_uncertainty(mean, covariance, key, *args, **kwargs):
        return monte_carlo(fun, trials)(mean, covariance, key, *args, **kwargs)

    return fun_with_uncertainty

from typing import Any, Callable, Dict
from jax import numpy as jnp
from jax import jacfwd, jacrev, vmap, jit, random, eval_shape
from scipy.io import savemat


def linear_uncertainty(fun: Callable):
    """See Measurements.jl in https://mitmath.github.io/18337/lecture19/uncertainty_programming"""

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

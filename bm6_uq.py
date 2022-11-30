# Import required packages
import h5py
import matplotlib.font_manager as fm
import numpy as np
# LUPROX
from jax import eval_shape, grad, jacfwd, jacrev, jit, lax, nn
from jax import numpy as jnp
from jax import random, value_and_grad, vmap
from jax.example_libraries import optimizers
from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import (DistributedTransducer, Domain, Medium, Sensors,
                            Sources, TimeAxis, _circ_mask, _points_on_circle)
from jwave.signal_processing import (analytic_signal, apply_ramp,
                                     gaussian_window, smooth)
from luprox import linear_uncertainty, mc_uncertainty
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.io import loadmat, savemat
from tqdm import tqdm

'''
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

        out_cov = (jnp.abs(J)**2) @ covariance # this factor of 4 is odd
        del J
        out_cov = jnp.reshape(out_cov, out_shape)
        out_mean = jnp.reshape(out_mean, out_shape)

        return out_mean, out_cov

    return jit(fun_with_uncertainty)




def monte_carlo(fun: Callable, trials):
    def sampling_function(mean, covariance, key):
        def _sample(mean, L, key, *args, **kwargs):
            noisy_x = mean + jnp.dot(L, random.normal(key, mean.shape))
            return fun(noisy_x, *args, **kwargs)

        mean = mean.real
        covariance = covariance.real
        keys = random.split(key, trials)

        L = jnp.linalg.cholesky(covariance)
        meanval = 0
        var = 0
        for i in range(trials):
            print(f"mc_mean_{i}")
            sample = _sample(mean, L, keys[i])
            meanval = meanval + sample/trials
            del sample

        for i in range(trials):
            print(f"mc_cov_{i}")
            sample = _sample(mean, L, keys[i])
            var = var + jnp.abs(sample-meanval)**2/trials
            del sample
        return meanval, var

    return sampling_function


def mc_uncertainty(fun: Callable, trials):
    def fun_with_uncertainty(mean, covariance, key):
        return monte_carlo(fun,
         trials)(mean, covariance, key)
    return fun_with_uncertainty
'''

# Settings
src_file = "SC1_coarse.mat"
sos_file = "sos.mat"
density_file = "density.mat"
skull_mask = "skull_mask.mat"

c0 = 1500.
f0 = 0.3e6
ppw = 6
cfl = 0.3
cov = (0.1)**2

c_mean = 2480.9
d_mean = 1782.5

# Slope values
alpha = 1.333
alpha_std = .1670
beta = 0 #104.9
beta_std = 18.7 #298.35

# c = alpha*HU + beta

# Load data
src_mat = loadmat(src_file)
sos_mat = loadmat(sos_file)
density_mat = loadmat(density_file)
skull_mat = loadmat(skull_mask)

# Extracting fields
src_bli = jnp.asarray(src_mat["src_field"], dtype=jnp.float32)[29:-29]
sound_speed = jnp.asarray(sos_mat["sound_speed"], dtype=jnp.float32)
density = jnp.asarray(density_mat["density"], dtype=jnp.float32)
skull_mask = jnp.asarray(skull_mat["skull_mask"], dtype=jnp.float32)
skull_density = skull_mask * density

# Prepare
skull_sos = sound_speed * skull_mask
bg_sos = sound_speed * (1 - skull_mask)

# Pad medium fields for PML
skull_sos = jnp.pad(skull_sos, ((20, 21), (20, 21), (20, 21)), mode="edge")
bg_sos = jnp.pad(bg_sos, ((20, 21), (20, 21), (20, 21)), mode="edge")
density = jnp.pad(density, ((20, 21), (20, 21), (20, 21)), mode="edge")
src_bli = jnp.pad(src_bli, ((10, 10), (10, 10), (10, 10)), mode="edge")
skull_density = jnp.pad(skull_density, ((20, 21), (20, 21), (20, 21)), mode="edge")
skull_mask = jnp.pad(skull_mask, ((20, 21), (20, 21), (20, 21)), mode="edge")
sound_speed = skull_sos + bg_sos

# Define domain
N = skull_sos.shape
c0 = c0
source_f0 = f0
ppw = ppw
dx = c0 / (ppw * source_f0)
dx = [dx]*len(N)

domain = Domain(N, dx)

# Make into fourier fields
src_bli = jnp.expand_dims(src_bli, -1)
sound_speed = FourierSeries(sound_speed, domain)
src_field = FourierSeries(src_bli, domain)
density = FourierSeries(density, domain)

# Define medium
medium =  Medium(domain=domain, sound_speed=sound_speed, pml_size=20)
time_axis = TimeAxis.from_medium(medium, cfl=cfl)
#time_axis.t_end = time_axis.t_end / 2
print(len(time_axis.to_array()))

# Define source term
source_mag = 1.
t = time_axis.to_array()
s1 = source_mag * jnp.sin(2 * jnp.pi * f0 * t)
signal = apply_ramp(s1, time_axis.dt, f0)

transducer = DistributedTransducer(
  mask = src_bli,
  signal = signal,
  dt = time_axis.dt,
  domain = Domain
)

# Define sensor plane
x = np.arange(282)
y = 182//2
z = np.arange(182)
x, y, z = np.meshgrid(x,y,z)
x = x.flatten()
y = y.flatten()
z = z.flatten()

sensors_positions = (x, y, z)
sensors = Sensors(positions=sensors_positions)

# Define simulation function
@jit
def simulate_sos(sound_speed):
    medium = Medium(
        domain=domain,
        sound_speed=sound_speed,
        density=density,
        pml_size=20.
    )
    return simulate_wave_propagation(
      medium,
      time_axis,
      sources = transducer,
      sensors = sensors,
      checkpoint=True)

@jit
def get_field(errors, skull_density, skull_mask, bg_sos):
    alpha, beta = errors
    skull_density = skull_density - d_mean
    skull_sos = skull_mask*(skull_density * alpha + beta + c_mean)
    sos = skull_sos + bg_sos
    sos = FourierSeries(sos, domain)
    p = simulate_sos(sos)
    p = np.reshape(p, (-1, 282, 182))
    p_max = jnp.max(jnp.abs(p[1400:])**2, axis=0)
    return jnp.sqrt(p_max)

get_field_lup = linear_uncertainty(get_field)
covariance = jnp.array([[alpha_std, 0], [0, beta_std]])**2

# Calculate uncertainty
x = jnp.array([alpha, beta])
mu_linear, cov_linear = get_field_lup(x, covariance, skull_density, skull_mask, bg_sos)
cov_linear = cov_linear *jnp.sqrt(2)
print(mu_linear)

# Monte carlo covariance
key = random.PRNGKey(42)
get_field_mc = mc_uncertainty(get_field, 10)
mu_mc, cov_mc = get_field_mc(x, covariance, key, skull_density, skull_mask, bg_sos)

# Make figure
fig, ax = plt.subplots(2, 2, figsize=(7,6), dpi=100)

im1 = ax[0,0].imshow(medium.sound_speed.on_grid[:,91,:], cmap="inferno", vmin=1500)
cbar = fig.colorbar(im1, ax=ax[0,0])
ax[0,0].legend(loc="lower right")
ax[0,0].set_title("Speed of sound")
ax[0,0].set_axis_off()

im1 = ax[0,1].imshow(mu_linear, cmap="inferno", vmin=0)
cbar = fig.colorbar(im1, ax=ax[0,1])
ax[0,1].axis('off')
ax[0,1].set_title('Unperturbed solution')

# Scale bar
fontprops = fm.FontProperties(size=12)
scalebar = AnchoredSizeBar(
    ax[0,1].transData,
    25, '1 cm', 'lower right',
    pad=0.3,
    color='white',
    frameon=False,
    size_vertical=2,
    fontproperties=fontprops)
ax[0,1].add_artist(scalebar)

im1 = ax[1,0].imshow(cov_mc, cmap="inferno", vmin=0)#, vmax=1.)
cbar = fig.colorbar(im1, ax=ax[1,0])
ax[1,0].axis('off')
ax[1,0].set_title('MC untertainty (N=100)')

im1 = ax[1,1].imshow(cov_linear, cmap="inferno", vmin=0)#, vmax=1.)
cbar = fig.colorbar(im1, ax=ax[1,1])
ax[1,1].axis('off')
ax[1,1].set_title('Linear uncertainty propagation')

fig.tight_layout()

# Save figure
plt.savefig("pressure_uq.png")
plt.close()

# Save fields
mdic = {
  "pressure": mu_linear,
  "cov": cov_linear,
  "cov_mc": cov_mc
}
savemat("uq_large.mat", mdic)

## Plane trough the focus
# Save x-y and x-z

## Import required packages
import matplotlib.font_manager as fm
import numpy as np
from jax import numpy as jnp
from jax import random, jit
from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import (
    DistributedTransducer,
    Domain,
    Medium,
    Sensors,
    TimeAxis,
)
from jwave.signal_processing import apply_ramp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.io import loadmat, savemat

from luprox import linear_uncertainty, mc_uncertainty

## Settings
src_file = "matfiles/SC1_coarse.mat"
bg_sos_file = "matfiles/background_sound_speed.mat"
density_file = "matfiles/density.mat"
skull_mask = "matfiles/skull_mask.mat"

c0 = 1500.0
f0 = 0.5e6
ppw = 6
cfl = 0.3
dx = 0.0005

# Previously evaluated in MATLAB
c_mean = 2480.9
d_mean = 1782.5

alpha = 1.333  # Slope values
alpha_std = 0.168  # Slope std
beta = 0  # Beta value
beta_std = 18.8  # Beta std

# Load data
src_mat = loadmat(src_file)
bg_sos_mat = loadmat(bg_sos_file)
density_mat = loadmat(density_file)
skull_mat = loadmat(skull_mask)

# Extracting fields
src_bli = jnp.asarray(src_mat["src_field"], dtype=jnp.float32)[
    29:-29
]  # Trim as source has different size
bg_sound_speed = jnp.asarray(bg_sos_mat["sound_speed"], dtype=jnp.float32)
density = jnp.asarray(density_mat["density"], dtype=jnp.float32)
skull_mask = jnp.asarray(skull_mat["skull_mask"], dtype=jnp.float32)
skull_density = skull_mask * density

# Prepare background sound speed (remove skull region)
bg_sos = bg_sound_speed * (1 - skull_mask)

# Pad medium fields for PML
bg_sound_speed = jnp.pad(bg_sound_speed, ((20, 21), (20, 21), (20, 21)), mode="edge")
bg_sos = jnp.pad(bg_sos, ((20, 21), (20, 21), (20, 21)), mode="edge")
density = jnp.pad(density, ((20, 21), (20, 21), (20, 21)), mode="edge")
src_bli = jnp.pad(src_bli, ((10, 10), (10, 10), (10, 10)), mode="edge")
skull_density = jnp.pad(skull_density, ((20, 21), (20, 21), (20, 21)), mode="edge")
skull_mask = jnp.pad(skull_mask, ((20, 21), (20, 21), (20, 21)), mode="edge")

# Define domain
N = bg_sound_speed.shape
source_f0 = f0
dx = [dx] * len(N)

domain = Domain(N, dx)

# Make into fourier fields
src_bli = jnp.expand_dims(src_bli, -1)
bg_sound_speed = FourierSeries(bg_sound_speed, domain)
src_field = FourierSeries(src_bli, domain)
density = FourierSeries(density, domain)

# Define medium
medium = Medium(domain=domain, sound_speed=bg_sound_speed, pml_size=20)
time_axis = TimeAxis.from_medium(medium, cfl=cfl)

# Define source term
source_mag = 1.0
t = time_axis.to_array()
s1 = source_mag * jnp.sin(2 * jnp.pi * f0 * t)
signal = apply_ramp(s1, time_axis.dt, f0)

transducer = DistributedTransducer(
    mask=src_bli, signal=signal, dt=time_axis.dt, domain=Domain
)

# Define sensor plane
x = np.arange(282)
y = 182 // 2
z = np.arange(182)
x, y, z = np.meshgrid(x, y, z)
x = x.flatten()
y = y.flatten()
z = z.flatten()

sensors = Sensors(positions=(x, y, z))

# Define simulation function
@jit
def simulate_sos(sound_speed):
    medium = Medium(
        domain=domain, sound_speed=sound_speed, density=density, pml_size=20.0
    )
    return simulate_wave_propagation(
        medium, time_axis, sources=transducer, sensors=sensors, checkpoint=True
    )


@jit
def get_field(values, skull_density, skull_mask, bg_sos):
    # Extract the parameters
    alpha, beta = values

    # Center skull density
    skull_density = skull_density - d_mean

    # Apply the linear model
    skull_sos = skull_mask * (skull_density * alpha + beta + c_mean)

    # Add background sound speed
    sos = skull_sos + bg_sos

    # Make it a field and simulate
    sos = FourierSeries(sos, domain)
    p = simulate_sos(sos)

    # Reshape output into a sequence of planes
    p = np.reshape(p, (-1, 282, 182))

    # Take the maximum after the initial transient
    p_max = jnp.max(jnp.abs(p[1400:]), axis=0)
    return p_max


# Test the function (and compiles it)
pressure = get_field(jnp.asarray([alpha, beta]), skull_density, skull_mask, bg_sos)

# Define mean and covariance
covariance = jnp.array([[alpha_std, 0], [0, beta_std]]) ** 2
x = jnp.array([alpha, beta])

# Transform function using linear uncertainty propagation
get_field_lup = linear_uncertainty(get_field)

# Calculate linear uncertainty
mu_linear, cov_linear = get_field_lup(x, covariance, skull_density, skull_mask, bg_sos)

# Do the same using Monte Carlo (200 runs)
key = random.PRNGKey(43)
get_field_mc = mc_uncertainty(get_field, 200)
mu_mc, cov_mc = get_field_mc(x, covariance, key, skull_density, skull_mask, bg_sos)

# ---------------------------

# Make figure
fig, ax = plt.subplots(2, 2, figsize=(7, 6), dpi=400)

im1 = ax[0, 0].imshow(medium.sound_speed.on_grid[:, 91, :], cmap="inferno", vmin=1500)
cbar = fig.colorbar(im1, ax=ax[0, 0])
ax[0, 0].legend(loc="lower right")
ax[0, 0].set_title("Speed of sound")
ax[0, 0].set_axis_off()

im1 = ax[0, 1].imshow(mu_linear, cmap="inferno", vmin=0)
cbar = fig.colorbar(im1, ax=ax[0, 1])
ax[0, 1].axis("off")
ax[0, 1].set_title("Unperturbed solution")

# Scale bar
fontprops = fm.FontProperties(size=12)
scalebar = AnchoredSizeBar(
    ax[0, 1].transData,
    25,
    "1 cm",
    "lower right",
    pad=0.3,
    color="white",
    frameon=False,
    size_vertical=2,
    fontproperties=fontprops,
)
ax[0, 1].add_artist(scalebar)

im1 = ax[1, 0].imshow(cov_mc, cmap="inferno", vmin=0)  # , vmax=1.)
cbar = fig.colorbar(im1, ax=ax[1, 0])
ax[1, 0].axis("off")
ax[1, 0].set_title("MC untertainty (N=100)")

im1 = ax[1, 1].imshow(cov_linear, cmap="inferno", vmin=0)  # , vmax=1.)
cbar = fig.colorbar(im1, ax=ax[1, 1])
ax[1, 1].axis("off")
ax[1, 1].set_title("Linear uncertainty propagation")

fig.tight_layout()

# Save figure
plt.savefig("pressure_uq.png")
plt.close()

# Save fields
mdic = {"pressure": mu_linear, "cov": cov_linear, "cov_mc": cov_mc}
savemat("mc_500k/uq_large.mat", mdic)

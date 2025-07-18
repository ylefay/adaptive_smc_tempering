# Generate random standard deviations
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(1)
n_dim = 3
key, subkey1, subkey2 = jax.random.split(key, 3)
std = jax.random.uniform(subkey1, shape=(n_dim,), minval=0.0, maxval=5.0)

# Generate random correlations
corr = jax.random.uniform(subkey2, shape=(int(n_dim * (n_dim - 1) / 2),), minval=-1.0, maxval=1.0)

# Create std_m matrix (diagonal matrix with std on the diagonal)
std_m = jnp.eye(n_dim) * std

# Create correlation matrix
corr_m = jnp.ones((n_dim, n_dim))
corr_m = corr_m.at[jnp.triu_indices(n_dim, k=1)].set(corr)
corr_m = corr_m.at[jnp.tril_indices(n_dim, k=-1)].set(corr)

# Compute covariance matrix
cov_m = std_m @ corr_m @ std_m / 12
print(cov_m)
print(jax.scipy.linalg.eigh(cov_m))

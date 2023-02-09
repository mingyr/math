import jax.random as random
import jax.numpy as jnp
def pca(X, l):
  n, m = X.shape # Data matrix X, assumes 0-centered
  C = jnp.matmul(X.T, X) / (n-1) # Compute covariance matrix
  eigen_vals, eigen_vecs = jnp.linalg.eig(C) # Eigen decomposition 
  eigen_vecs = jnp.real(eigen_vecs[:, :l]) # Cannot be complex number
  X_pca = jnp.matmul(X, eigen_vecs) # Project X onto PC space
  return X_pca

def pca_dec(X, X_pca, l):
  n, m = X.shape # Data matrix X, assumes 0-centered
  C = jnp.matmul(X.T, X) / (n-1) # Compute covariance matrix
  eigen_vals, eigen_vecs = jnp.linalg.eig(C) # Eigen decomposition 
  eigen_vecs = jnp.real(eigen_vecs[:, :l]) # Cannot be complex number
  X_dec = jnp.matmul(X_pca, eigen_vecs.T) # Project X onto PC space
  return X_dec

A = random.normal(random.PRNGKey(0), (3, 4))
print("A: {}".format(A))
A_offset = A.mean(axis=0)
A = A - A_offset
A_pca = pca(A, 2) 
print("A_pca: {}".format(A_pca))
A_dec = pca_dec(A, A_pca, 2)
A_dec = A_dec + A_offset
print("Offset_A: {}".format(A_dec))


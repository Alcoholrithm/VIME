import numpy as np


def mask_generator (p_m, x):
  """Generate mask vector.
  
  Args:
    - p_m: corruption probability
    - x: feature matrix
    
  Returns:
    - mask: binary mask matrix 
  """
  mask = np.random.binomial(1, p_m, x.shape)

  return np.expand_dims(mask, axis=0)
  
def pretext_generator (m, x, empirical_dist):  
  """Generate corrupted samples.
  
  Args:
    m: mask matrix
    x: feature matrix
    
  Returns:
    m_new: final mask matrix after corruption
    x_tilde: corrupted feature matrix
  """
  
  # Parameters
  dim = x.shape[0]
  # Randomly (and column-wise) shuffle data
  x_bar = np.zeros([1, dim])

  rand_idx = np.random.randint(0, len(empirical_dist), size=dim)
  
  x_bar = np.array([empirical_dist[rand_idx[i], i] for i in range(dim)])
  
  # Corrupt samples
  x_tilde = x * (1-m) + x_bar * m  
  # Define new mask matrix
  m_new = 1 * (x != x_tilde)

  return m_new.squeeze(), x_tilde.squeeze()

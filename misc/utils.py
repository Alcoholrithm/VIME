"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

vime_utils.py
- Various utility functions for VIME framework

(1) mask_generator: Generate mask vector for self and semi-supervised learning
(2) pretext_generator: Generate corrupted samples for self and semi-supervised learning
(3) perf_metric: prediction performances in terms of AUROC or accuracy
(4) convert_matrix_to_vector: Convert two dimensional matrix into one dimensional vector
(5) convert_vector_to_matrix: Convert one dimensional vector into one dimensional matrix
"""

# Necessary packages
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


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


# def convert_matrix_to_vector(matrix):
#   """Convert two dimensional matrix into one dimensional vector
  
#   Args:
#     - matrix: two dimensional matrix
    
#   Returns:
#     - vector: one dimensional vector
#   """
#   # Parameters
#   no, dim = matrix.shape
#   # Define output  
#   vector = np.zeros([no,])
  
#   # Convert matrix to vector
#   for i in range(dim):
#     idx = np.where(matrix[:, i] == 1)
#     vector[idx] = i
    
#   return vector

# def convert_vector_to_matrix(vector):
#   """Convert one dimensional vector into two dimensional matrix
  
#   Args:
#     - vector: one dimensional vector
    
#   Returns:
#     - matrix: two dimensional matrix
#   """
#   # Parameters
#   no = len(vector)
#   dim = len(np.unique(vector))
#   # Define output
#   matrix = np.zeros([no,dim])
  
#   # Convert vector to matrix
#   for i in range(dim):
#     idx = np.where(vector == i)
#     matrix[idx, i] = 1
    
#   return matrix

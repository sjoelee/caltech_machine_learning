import random
import numpy as np
import matplotlib.pyplot as plt
import copy

def generate_dataset(num_of_points=10):
  X = np.ones((num_of_points, 3)) # these are 1 because of x0 = 1 for the constant term
  X[:, 1:] = np.random.uniform(-1, 1, (num_of_points, 2))
  return X

def classify(X, w, y=None):
  return np.sign(np.dot(X, w))

def get_average_num_iterations(num_runs=1, num_training_points=10, plot=False):
  num_iteration = np.zeros((num_runs, 1))
  index = 0

  run_number = 0
  while run_number < num_runs:
    num_iteration[index] = iterate(num_training_points, plot)
    index += 1
    run_number += 1

  return np.sum(num_iteration)/num_runs

def iterate(num_training_points=10, plot=False):
  # create target function and training set
  coeff = np.random.uniform(-1, 1, (3, 1))
  X = generate_dataset(num_training_points)
  y_true = classify(X, coeff)

  # initialize weights
  w = np.zeros((3, 1))
  num_iterations = 0

  while True:
    y = classify(X, w)
    error_indexes = copy.copy(np.where(y != y_true)[0]) # take first element because that contains the indices
  
    if len(error_indexes) > 0:
      np.random.shuffle(error_indexes)
      chosen_index = error_indexes[0]

      chosen_x = X[chosen_index,:]
      chosen_x = chosen_x.reshape(3,1) # need to reshape to create a column vector of these values
      w = w + y_true[chosen_index] * chosen_x
      
      num_iterations += 1
    else:
      print("CONVERGED at ", num_iterations)
      break

  if plot:
    plot_points(X, coeff, w)

  return num_iterations

def plot_points(X, coeff, w):
  x = np.arange(-2, 2, 0.2)
  plt.plot(x, (-coeff[1]*x - coeff[0])/coeff[2], '--')
  plt.plot(x, (-w[1]*x - w[0])/w[2], 'r--')
  y_true = classify(X, coeff)

  for num in xrange(len(y_true)):
    if y_true[num] > 0:
      plt.plot(X[num][1], X[num][2], 'r+')
    else:
      plt.plot(X[num][1], X[num][2], 'ro')

  plt.xlim(-1,1)
  plt.ylim(-1,1)
  plt.show()

  return

if __name__ == '__main__':
  average_num = get_average_num_iterations(1000, 10)
  print "AVERAGE # ", average_num
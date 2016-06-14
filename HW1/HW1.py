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

def get_average_num_iterations(num_runs=1, num_training_points=10, num_test_points=1000, plot=False):
  num_iteration = 0
  num_prob = 0
  index = 0

  while index < num_runs:
    num_iteration_index, num_prob_index = iterate(num_training_points, num_test_points, plot)
    num_iteration += num_iteration_index
    num_prob += num_prob_index
    index += 1

  return num_iteration/num_runs, num_prob/num_runs

def iterate(num_training_points, num_test_points, plot, w=None):
  # create target function and training set
  coeff = np.random.uniform(-1, 1, (3, 1))
  X = generate_dataset(num_training_points)
  y_true = classify(X, coeff)

  # initialize weights
  if w is None:
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
      break

  # test how well w classifies the training points
  X_test = generate_dataset(num_test_points)
  y_true = classify(X_test, coeff)
  y = classify(X_test, w)
  sample_prob = np.sum(y != y_true)/float(num_test_points)

  if plot:
    plot_points(X, coeff, w)

  return num_iterations, sample_prob

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
  average_num, prob_of_error = get_average_num_iterations(1000, 100, 1000)
  print "AVERAGE # ", average_num
  print "Probability of Error ", prob_of_error
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

def get_average_num_iterations(num_runs=1, num_training_points=10):
  num_iteration = np.zeros((num_runs, 1))
  index = 0

  run_number = 0
  while run_number < num_runs:
    num_iteration[index] = iterate(num_training_points)
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
    if num_iterations % 100 == 0:
      print "ITERATION #: ", num_iterations

    y = classify(X, w)
    error_indexes = copy.copy(np.where(y != y_true)[0]) # take first element because that contains the indices
  
    if len(error_indexes) > 0:
      np.random.shuffle(error_indexes)
      chosen_index = error_indexes[0]
      w = w + y_true[chosen_index] * X[chosen_index, :].T
      num_iterations += 1
    else:
      break

    if plot:
      plot_points(X, coeff, w)

  print num_iterations

  return num_iterations

def plot_points(X, coeff, w, y):
  return

def average_PLA_iterate(number_of_iterations=1, num_training_points=10):
  iterations = 0

  for iter in xrange(number_of_iterations):
    iterations += PLA_iterate(num_training_points)

  print("FINAL ANSWER ", iterations / number_of_iterations)

def PLA_iterate(num_training_points=10):
  num_iterations = 0
  num_misclassified_points = 0

  # create the target function
  f = (np.random.rand(2, 2) - 0.5) * 2

  # a = slope, b = (y-intercept)
  m = (f[0][1] - f[1][1]) / (f[0][0] - f[1][0])
  b = f[0][1] - (m * f[0][0])

  # initialize weights to be 0
  w = np.array([[0, 0, 0]])
  misclassified_points = np.array([[]])
  training_points = (np.random.rand(num_training_points, 2) - 0.5) * 2

  # used for plotting
  x = np.arange(-2, 2, 0.2)

  misclassified_points_index = []

  while True:
    num_iterations += 1
    if num_iterations % 100 == 0: 
      print("ITERATION # ", num_iterations)

    for num in xrange(num_training_points):
      # see if point is +/- based off of its relation to f
      true_class = 1 if (training_points[num][1] > m * training_points[num][0] + b) else -1

      # determine perceptron output 
      perceptron_class = int(np.sign(1 * w[0][0] + training_points[num][0] * w[0][1] + training_points[num][1] * w[0][2]))

      # see how our hypothesis classifies it based off of w
      if perceptron_class != true_class:
        misclassified_points_index.append(num)
        num_misclassified_points += 1

    if num_misclassified_points > 0:
      # pick a random misclassified point and update w. Then reset num_misclassified_points
      rand_index = random.randint(0, len(misclassified_points_index) - 1)
      chosen_index = misclassified_points_index[rand_index]
      true_class = 1 if (training_points[chosen_index][1] > m * training_points[chosen_index][0] + b) else -1

      x_hat = np.array([1])
      x_hat = np.append(x_hat, training_points[chosen_index,:])
      w = w + true_class * x_hat
      print ("w = ", w)
      # w = w / (np.linalg.norm(w)) # normalize w to prevent it from blowing up

      num_misclassified_points = 0
      misclassified_points_index = []
    else:
      break

    if (num_iterations > 500 and num_iterations < 510): # or num_iterations < 10:
      plt.plot(x, m * x + b)
      plt.plot(x, -w[0][0]/w[0][2] - w[0][1]/w[0][2] * x, 'r--')

      for num in xrange(num_training_points):
        if (training_points[num][1] > m * training_points[num][0] + b):
          plt.plot(training_points[num][0], training_points[num][1], 'r+')
        else:
          plt.plot(training_points[num][0], training_points[num][1], 'ro')
      plt.xlim(-1,1)
      plt.ylim(-1,1)
      plt.show()

  print("CONVERGED at ", num_iterations)
  print("f values: %.3f, %.3f, %.3f" % (m, -1, b))
  print("g values: %.3f, %.3f, %.3f" % (w[0][0], w[0][1], w[0][2]))

  # plot final graphs
  # plt.plot(x, m * x + b)
  # plt.plot(x, -w[0][0]/w[0][2] - w[0][1]/w[0][2] * x, 'r--')
  # for num in xrange(num_training_points):
  #   if (training_points[num][1] > m * training_points[num][0] + b):
  #     plt.plot(training_points[num][0], training_points[num][1], 'r+')
  #   else:
  #     plt.plot(training_points[num][0], training_points[num][1], 'ro')

  # plt.xlim(-1,1)
  # plt.ylim(-1,1)
  # plt.show()

  return num_iterations

if __name__ == '__main__':
  # average_PLA_iterate(1000, 10)
  get_average_num_iterations(10, 10)
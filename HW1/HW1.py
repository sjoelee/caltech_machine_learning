import random
import numpy as np
import matplotlib.pyplot as plt

def average_PLA_iterate(number_of_iterations=1 , num_training_points=10):
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
  average_PLA_iterate(1000, 10)
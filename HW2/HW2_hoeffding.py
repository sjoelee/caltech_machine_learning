import numpy as np
import matplotlib.pyplot as plt

def flip_coins(num_coins, num_flips):
    C = np.random.randint(2, size=(num_coins, num_flips))
    min_index = np.argmin(np.sum(C, axis=1))
    rand_index = np.random.randint(num_coins)
    # print min_index, rand_index
#     v_min = np.sum(C[min_index,:])/float(num_flips)
#     v_rand = np.sum(C[rand_index,:])/float(num_flips)
#     v_1 = np.sum(C[1,:])/float(num_flips)
    min_sum = np.sum(C[min_index,:])
    rand_sum = np.sum(C[rand_index,:])
    one_sum = np.sum(C[1,:])
    return min_sum, rand_sum, one_sum

def plot_histogram(range, weights, bins):
    plt.hist(range, weights=weights, bins=bins)
    plt.show()

def hoeffding_experiment(num_run_times, num_coins, num_flips):
    min_sum = 0
    rand_sum = 0
    one_sum = 0
    min_sum_hist = np.zeros([num_flips+1,1])
    rand_sum_hist = np.zeros([num_flips+1,1])
    one_sum_hist = np.zeros([num_flips+1,1])
    
    for run_num in xrange(num_run_times):
        min_sum_, rand_sum_, one_sum_ = flip_coins(num_coins, num_flips)
        rand_sum += rand_sum_
        one_sum += one_sum_
        min_sum_hist[min_sum_] += 1
        rand_sum_hist[rand_sum_] += 1
        one_sum_hist[one_sum_] += 1
    
    plot_histogram(range(0,11), weights=rand_sum_hist, bins=11)
    plot_histogram(range(0,11), weights=min_sum_hist, bins=11)
    plot_histogram(range(0,11), weights=one_sum_hist, bins=11)
    
    print "min average: ", min_sum/(float(num_run_times) * num_flips)
    print "one average: ", one_sum/(float(num_run_times) * num_flips)
    print "rand average: ", rand_sum/(float(num_run_times) * num_flips)

if __name__ == '__main__':
    num_coins = 1000
    num_flips = 10
    num_run_times = 100000
    min_sum = 0

    hoeffding_experiment(num_run_times, num_coins, num_flips)
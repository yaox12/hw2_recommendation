#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import time
import numpy as np
import scipy.spatial.distance as dis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

k = 50
lmd = 0.01
alpha = 0.01

def load_train_data(filename):
    '''
    Load train data from file:
    Column: user ID, movie ID, score, date.

    Return: Numpy matrix of score: user * movie,
            indicate matrix, value 1 when exists.
    '''
    score = []
    user_id = {}
    indicate = np.zeros((10000, 10000))
    with open(filename, 'r') as infile:
        t_user, user_num = -1, -1
        row = np.zeros(10000)
        # linenum = 0
        for line in infile:
            # linenum += 1
            # if linenum % 100000 == 0:
            #     print(linenum)

            content = line.strip().split(' ')
            line_user, line_movie, line_score = int(content[0]), int(content[1]), int(content[2]) 
            if line_user != t_user:
                if t_user != -1:
                    score.append(row)
                t_user = line_user
                user_num += 1
                user_id[t_user] = user_num
                row = np.zeros(10000)
            row[line_movie - 1] = line_score
            indicate[user_num][line_movie - 1] = 1
        score.append(row)
        infile.close()
        assert(user_num == 9999)
    return np.array(score), user_id, indicate

def load_test_data(user_id, filename):
    test_set = {}
    with open(filename, 'r') as infile:
        for line in infile:
            content = line.strip().split(' ')
            line_user, line_movie, line_score = int(content[0]), int(content[1]), int(content[2])
            test_set[(user_id[line_user], line_movie - 1)] = line_score
        infile.close()
    
    return test_set

def rmse(u, v, test_set):
    predictions, targets = [], []
    pre_matrix = np.dot(u, v.T)
    for item in test_set:
        targets.append(test_set[item])
        predictions.append(pre_matrix[item[0]][item[1]])
    
    predictions = np.array(predictions, dtype = np.float)
    targets = np.array(targets, dtype = np.float)
    return np.sqrt(np.mean((predictions - targets)**2))

def plot_fig(y, title):
    plt.figure(0)
    plt.clf()
    x = [i for i in range(len(y))]
    plt.plot(x, y, '--*k')
    plt.title(title)
    plt.savefig(title + '.png')

if __name__ == '__main__':
    begin = time.time()
    score, user_id, indicate = load_train_data(r'./data/netflix_train.txt')
    test_set = load_test_data(user_id, r'./data/netflix_test.txt')
    end = time.time()
    print('Time spent on loading data: %f seconds\n' % (end - begin))

    begin = time.time()
    u = np.random.random((10000, k))
    v = np.random.random((10000, k))
    target_vec, rmse_vec = [], []
    hadamard_func = lambda indicate, u, v, score: indicate * (np.dot(u, v.T) - score)
    last_j = 0
    while True:
        h_product = hadamard_func(indicate, u, v, score)
        j = 0.5 * np.linalg.norm(h_product) ** 2 + lmd * (np.linalg.norm(u) ** 2 + np.linalg.norm(v) ** 2)
        if j > last_j and last_j != 0:
            break
        last_j = j
        print('J: %f' % j)
        target_vec.append(j)
        rmse_now = rmse(u, v, test_set)
        print('RMSE: %f' % rmse_now)
        rmse_vec.append(rmse_now)
        
        gradient_u = np.dot(h_product, v) + 2 * lmd * u
        gradient_v = np.dot(h_product.T, u) + 2 * lmd * v
        u -= alpha * u
        v -= alpha * v
    end = time.time()
    print('Time spent on iteration: %f seconds\n' % (end - begin))

    plot_fig(target_vec, 'ObjectiveFunction')
    plot_fig(rmse_vec, 'RMSE')

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
import matplotlib.pyplot as plt

def load_score_data(filename):
    '''
    Load train or test data from file:
    Column: user ID, movie ID, score, date.

    Return: Numpy matrix of score to user * movie.
    '''
    score = []
    user_id = {}
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
        score.append(row)
        infile.close()
        assert(user_num == 9999)
    return np.array(score), user_id


def calc_sim(score):
    '''
    Return： Numpy matrix of similarity between users.
    sim[i][j] is the cosine similarity between user i and j.
    '''
    sim = np.dot(score, score.T)
    nvec = []
    for vec in score:
        nvec.append(np.linalg.norm(vec))
    assert(len(nvec) == 10000)
    nvec = np.array(nvec).reshape(10000, 1)
    # print(nvec.shape)
    nmatrix = np.dot(nvec, nvec.T)
    # print(nmatrix.shape)
    return sim / nmatrix


def predict(score, sim, user_id, filename):
    '''
    Return: RMSE of test data and prediction data.
    '''
    predictions, targets = [], []
    sum_sim = np.sum(sim, axis=1)
    linenum = 0
    with open(filename, 'r') as infile:
        for line in infile:
            linenum += 1
            if linenum % 100000 == 0:
                print(linenum)

            content = line.strip().split(' ')
            line_user, line_movie, line_score = int(content[0]), int(content[1]), int(content[2])
            targets.append(line_score)
            predictions.append(np.dot(sim[user_id[line_user]], score[:,line_movie - 1]) / sum_sim[user_id[line_user]])
        infile.close()
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(np.mean((predictions - targets)**2))


if __name__ == '__main__':
    begin = time.time()
    score, user_id = load_score_data(r'./data/netflix_train.txt')
    end = time.time()
    print('Time spent on loading train set: %f seconds\n' % (end - begin))

    begin = time.time()
    # sim = 1 - dis.squareform(dis.pdist(score, 'cosine'))
    sim = calc_sim(score)
    end = time.time()
    print('Time spent on calculating sim matrix: %f seconds\n' % (end - begin))

    begin = time.time()
    rmse = predict(score, sim, user_id, r'./data/netflix_test.txt')
    end = time.time()
    print(rmse)
    print('Time spent on predition: %f seconds\n' % (end - begin))

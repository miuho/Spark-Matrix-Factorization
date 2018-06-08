#!/usr/bin/env python3

# HingOn Miu (hmiu)


import sys
import numpy as np
import os
import random
import time

from pyspark import SparkConf, SparkContext


num_partitions = 19 * 4
N = num_partitions
x_block_dim = 0
y_block_dim = 0
step = 0.0


# parse elements in dataset
def map_line_to_elements(line):
    tokens = line.split(",")
    return int(tokens[0]), int(tokens[1]), float(tokens[2])


# parse elements to correct format in dataset
def map_line_to_ij_line(line):
    tokens = line.split(",")
    x_id = int(tokens[0])
    y_id = int(tokens[1])
    x_block_id = int(x_id / x_block_dim)
    y_block_id = int(y_id / y_block_dim)
    #print (x_block_id, y_block_id, x_id, y_id)
    #tmp_files[x_block_id][y_block_id].write(line)
    return (str(x_block_id) + ' ' + str(y_block_id)), (tokens[0] + "," + tokens[1] + "," + str(float(tokens[2])) + ";")


# concat strings
def reduce_line_to_lines(a, b):
    return a + b


# blockify the data matrix
def blockify_data(sc, csv_file, N):
    global x_block_dim
    global y_block_dim
    
    scores = sc.textFile(csv_file).map(map_line_to_elements).cache()
    max_x_id = scores.map(lambda x: x[0]).max()
    max_y_id = scores.map(lambda x: x[1]).max()
    scores.unpersist()

    # assume the id starts from 0
    x_block_dim = int((max_x_id + N) / N)
    y_block_dim = int((max_y_id + N) / N)

    # i j -> lines
    dataset = sc.textFile(csv_file, num_partitions).cache()
    
    tmp = dataset.map(map_line_to_ij_line) \
                    .reduceByKey(reduce_line_to_lines, 19).collect()
    dataset.unpersist()
    
    # broadcast variable to reduce network shuffle I/O
    data_blocks = sc.broadcast(tmp)
    
    return data_blocks


# one line is a partition of the factor matrix
def parse_factor_matrix_line(line):
    tokens = line.split(";")
    rows = []
    for token in tokens:
        row = []
        token = token.strip()
        if token == "":
            continue
        row_entries = token.split(",")
        for row_entry in row_entries:
            if row_entry == "":
                continue
            row.append(float(row_entry))
        rows.append(np.array(row))
    return rows


# randomly initialize the factor matrices
def initialize_factor_matrices(sc, N, K):
    random.seed(1) # always use the same seed to get deterministic results
    
    # line = 0 .. N
    # row = 0 .. x_block_dim
    # K random numbers in one row eg. 42,3,5;
    # x_block_dim rows in one line eg. 22,41,2;42,3,5;
    # N lines in w eg. [22,41,2;42,3,5;]
    
    w = []
    for i in range(0, N):
        line = ""
        for j in range(0, x_block_dim):
            row = ""
            for k in range(0, K):
                row += str(random.random()) + ","
            line += row +";"
        w += [(str(i), parse_factor_matrix_line(line))]
    
    all_W_rows = sc.parallelize(w, num_partitions).cache()
    # broadcast variable to reduce network shuffle I/O
    all_W_rows_bc = sc.broadcast(w)

    h = []
    for i in range(0, N):
        line = ""
        for j in range(0, y_block_dim):
            row = ""
            for k in range(0, K):
                row += str(random.random()) + ","
            line += row +";"
        h += [(str(i), parse_factor_matrix_line(line))]
    
    all_H_rows = sc.parallelize(h, num_partitions).cache()
    # broadcast variable to reduce network shuffle I/O
    all_H_rows_bc = sc.broadcast(h)

    return all_W_rows, all_H_rows, all_W_rows_bc, all_H_rows_bc


# only output changes to H rows
def sgd_on_one_block_H(elem):
    i_j, ((W_rows, H_rows), data_line) = elem
    [i, j] = i_j.split(' ')
    if (data_line == ""):
        return (j, H_rows)
    
    W_rows_offset = int(i) * x_block_dim
    H_rows_offset = int(j) * y_block_dim
    
    data_line = data_line.strip()
    data_samples = data_line.split(";")

    for data_sample in data_samples:
        if data_sample == "":
            continue
        tokens = data_sample.split(",")
        x_id = int(tokens[0])
        y_id = int(tokens[1])
        rating = float(tokens[2])

        diff = rating - np.dot(W_rows[x_id - W_rows_offset], H_rows[y_id - H_rows_offset])
        W_gradient = -2 * diff * H_rows[y_id - H_rows_offset]
        W_rows[x_id - W_rows_offset] -= step * W_gradient

        H_gradient = -2 * diff * W_rows[x_id - W_rows_offset]
        H_rows[y_id - H_rows_offset] -= step * H_gradient
    
    return (j, H_rows)


# only output changes to W rows
def sgd_on_one_block_W(elem):
    i_j, ((W_rows, H_rows), data_line) = elem
    [i, j] = i_j.split(' ')
    if (data_line == ""):
        return (i, W_rows)
    
    W_rows_offset = int(i) * x_block_dim
    H_rows_offset = int(j) * y_block_dim
    
    data_line = data_line.strip()
    data_samples = data_line.split(";")

    for data_sample in data_samples:
        if data_sample == "":
            continue
        tokens = data_sample.split(",")
        x_id = int(tokens[0])
        y_id = int(tokens[1])
        rating = float(tokens[2])

        diff = rating - np.dot(W_rows[x_id - W_rows_offset], H_rows[y_id - H_rows_offset])
        W_gradient = -2 * diff * H_rows[y_id - H_rows_offset]
        W_rows[x_id - W_rows_offset] -= step * W_gradient

        H_gradient = -2 * diff * W_rows[x_id - W_rows_offset]
        H_rows[y_id - H_rows_offset] -= step * H_gradient
    
    return (i, W_rows)


# outputs W/H rows to string
def factor_matrix_rows_to_string(elem):
    k, rows = elem
    line = ""
    for row in rows:
        for num in np.nditer(row):
            line += str(num) + ","
        line = line[:-1]
        line += "\n"
    line = line[:-1]
    #return (int(k), line)
    return line


# get elements count
def map_count(elem):
    a, rows = elem
    return len(rows)


# perform the SGD algorithm one block at a time
def sgd_block_by_block(sc, it, N, step_size, x_block_dim, y_block_dim, data_blocks, all_W_rows, all_H_rows, all_W_rows_bc, all_H_rows_bc):
    global step
    step = step_size
    
    def join_data_blocks(elem):
        i_j_1 , (w, h) = elem
        for data in data_blocks.value:
            i_j_2, line = data
            if (i_j_1 == i_j_2):
                return i_j_1, ((w, h), line)
        return i_j_1, ((w, h), "")
    
    new_all_H_rows = all_H_rows
    for i in range(0, N):
        i_str = str(i)
        
        W_rows = []
        for a in all_W_rows_bc.value:
            i_str_2, r = a
            if (i_str == i_str_2):
                W_rows = r
                break
        if (W_rows == []):
            W_rows = all_W_rows.lookup(i_str)[0]
    
        WH_rows = new_all_H_rows.map(lambda (k, v): (i_str + ' ' + k, (W_rows, v))).cache()
        if (i != 0):
            new_all_H_rows.unpersist()
        new_all_H_rows = WH_rows.map(join_data_blocks).map(sgd_on_one_block_H).cache()
        WH_rows.unpersist()
    
    new_all_W_rows = all_W_rows
    for j in range(0, N):
        j_str = str(j)
        
        H_rows = []
        for a in all_H_rows_bc.value:
            j_str_2, r = a
            if (j_str == j_str_2):
                H_rows = r
                break
        if (H_rows == []):
            H_rows = all_H_rows.lookup(j_str)[0]
    
        WH_rows = new_all_W_rows.map(lambda (k, v): (k + ' ' + j_str, (v, H_rows))).cache()
        if (j != 0):
            new_all_W_rows.unpersist()
        new_all_W_rows = WH_rows.map(join_data_blocks).map(sgd_on_one_block_W).cache()
        WH_rows.unpersist()
    
    all_W_rows.unpersist()
    all_H_rows.unpersist()
    
    if (it != 3):
        all_W_rows_bc.unpersist()
        all_H_rows_bc.unpersist()
        new_all_W_rows_bc = sc.broadcast(new_all_W_rows.collect())
        new_all_H_rows_bc = sc.broadcast(new_all_H_rows.collect())
    else:
        new_all_W_rows_bc = all_W_rows_bc
        new_all_H_rows_bc = all_H_rows_bc
    
    return new_all_W_rows, new_all_H_rows, new_all_W_rows_bc, new_all_H_rows_bc


# evaluate the current model on one block of the data
def evaluate_on_one_block(elem):
    i_j, ((W_rows, H_rows), data_line) = elem
    [i, j] = i_j.split(' ')
    W_rows_offset = int(i) * x_block_dim
    H_rows_offset = int(j) * y_block_dim
    
    data_line = data_line.strip()
    data_samples = data_line.split(";")
    error = .0
    n = 0

    for data_sample in data_samples:
        if data_sample == "":
            continue
        tokens = data_sample.split(",")
        x_id = int(tokens[0])
        y_id = int(tokens[1])
        rating = float(tokens[2])

        diff = rating - np.dot(W_rows[x_id - W_rows_offset], H_rows[y_id - H_rows_offset])
        error += diff ** 2
        n += 1

    return error, n


# perform evaluation of the model one block at a time
def evaluate_block_by_block(N, x_block_dim, y_block_dim, data_blocks, all_W_rows, all_H_rows):
    
    WH_rows = all_W_rows.cartesian(all_H_rows).map(lambda ((a1,a2),(b1,b2)): (a1 + ' ' + b1, (a2, b2))).cache()
    
    error_n = WH_rows.join(data_blocks).map(evaluate_on_one_block).cache()
    WH_rows.unpersist()
    
    error_total = error_n.map(lambda (a,b): a).reduce(lambda x, y: x + y)
    n_total = error_n.map(lambda (a,b): b).reduce(lambda x, y: x + y)
    error_n.unpersist()

    return error_total, n_total


if __name__ == "__main__":
    print ("Setup begins...")
    conf = SparkConf()
    #conf.setMaster("local")
    conf.setAppName("Matrix Factorization")
    #conf.set("spark.master", "local[*]")
    conf.set("spark.driver.memory", "15g")
    conf.set("spark.executor.memory", "15g")
    conf.set("spark.executor.cores", "4")
    conf.set("spark.python.worker.memory", "7g")
    #conf.set("spark.memory.fraction", "0.9")
    conf.set("spark.broadcast.blockSize", "64m")
    conf.set("spark.akka.frameSize", "2047") # enlarges task size
    conf.set("spark.driver.maxResultSize", "4g")
    conf.set("spark.network.timeout", "1200s")
    sc = SparkContext(conf = conf)
   
    # command line input
    #--executor-memory 15g --driver-memory 15g --driver-cores 4 --num-executors 19 --executor-cores 4

    print ("Out-of-Core SGD Matrix Factorization begins...")
    csv_file = sys.argv[1] #hdfs dataset location
    K = int(sys.argv[2]) #rank
    w_file = sys.argv[3] #hdfs w location
    h_file = sys.argv[4] #hdfs h location
    
    num_iterations = 4 # converages at this iteration
    eta = 0.001
    eta_decay = 0.99

    data_blocks = blockify_data(sc, csv_file, N)
    print("Done partitioning data matrix...")
    
    all_W_rows, all_H_rows, all_W_rows_bc, all_H_rows_bc = initialize_factor_matrices(sc, N, K)
    print("Done initializing factor matrices...")
    
    print("Start Stochastic Gradient Descent...")
    
    print("iteration", " seconds", " squared_error", " RMSE")
    #t1 = time.clock()
    for i in range(0, num_iterations):
        all_W_rows, all_H_rows, all_W_rows_bc, all_H_rows_bc = sgd_block_by_block(sc, i, N, eta, x_block_dim, y_block_dim, data_blocks, all_W_rows, all_H_rows, all_W_rows_bc, all_H_rows_bc)
        eta *= eta_decay
        
        print("___________" + str(i) + "_____________")
        # skip unnessary evaluation step
        #error, n = evaluate_block_by_block(N, x_block_dim, y_block_dim, data_blocks, all_W_rows, all_H_rows)
        #t2 = time.clock()
        #print (i, int(t2 - t1), error, np.sqrt(error / n))

    data_blocks.unpersist()
    all_W_rows_bc.unpersist()
    all_H_rows_bc.unpersist()
    
    all_W_rows.map(factor_matrix_rows_to_string).coalesce(1, shuffle=True).saveAsTextFile(w_file)
    all_W_rows.unpersist()

    all_H_rows.map(factor_matrix_rows_to_string).coalesce(1, shuffle=True).saveAsTextFile(h_file)
    all_H_rows.unpersist()

    #print("Stochastic Gradient Descent Done, computation time =", int(t2 - t1), "seconds, exit now")

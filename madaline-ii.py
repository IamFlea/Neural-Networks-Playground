# Many Adaptive Linear Neurons -  Rule 1   MR-I
from dataset import dataset_bipolar_output
from copy import copy
from math import exp
from functools import reduce

import random
print("NOT WORKING")
exit()
# Load data set
train, test = dataset_bipolar_output('datasets/sonar.all-data')
train = [[1,1], [0,0], [1,0], [0,1]],  [[1],[1],[-1],[-1]]

inputs = train[0]
inputs = list(map(lambda x: x+[1], inputs))
ref_output = list(map(lambda x: x[0], train[1]))
layers = [2, 1]

training_vectors = list(zip(inputs, ref_output))
learning_coef = 0.05
theta = 5
weights = [[0 for _ in inputs[0]] for _ in range(layers[0])]
weights_out = [0 for _ in range(layers[0])]+[0]
for epoch in range(700):
    hit = 0
    for inputs, desired_output in training_vectors:
        u_layer_1 = [sum(map(lambda x,y: x*y, inputs, neuron_w)) for neuron_w in weights]
        y_layer_1 = list(map(lambda x: 1 if x > 0 else -1, u_layer_1))+[1]
        u_layer_2 = sum(map(lambda x,y: x*y, y_layer_1, weights_out))
        output = 1 if u_layer_2 > 0 else -1
        #-----
        error = (desired_output - output) * learning_coef 
        if not error: 
            hit += 1
            continue

        # Iterate through first adalines
        adalines = enumerate(u_layer_1)
        adalines = filter(lambda x: abs(x[1]) < theta, adalines)
        if not adalines:
            continue
        # sort them 
        adalines = sorted(adalines, key=lambda x: x[1], reverse=desired_output==1)

        for idx, u in adalines: 
            w_old = weights[idx]
            w_new = map(sum, zip(list(map(lambda x, e=error: x*e, inputs)), w_old))
            weights[idx] = list(w_new)
            _u_layer = [sum(map(lambda x,y: x*y, inputs, neuron_w)) for neuron_w in weights]
            
            output = 1 if output else -1
            if output == desired_output:
                break

            w_old = weights[idx]
            w_new = map(sum, zip(list(map(lambda x, e=error: x*e, inputs)), w_old))
            u_layer_1 = [sum(map(lambda x,y: x*y, inputs, neuron_w)) for neuron_w in weights]
            y_layer_1 = list(map(lambda x: 1 if x > 0 else -1, u_layer_1))+[1]
            u_layer_2 = sum(map(lambda x,y: x*y, y_layer_1, weights_out))
            output = 1 if u_layer_2 > 0 else -1
            
        # MAKE ADALINE HERE

    print(hit)
print(f"Found {hit} / {len(train)} = {int(hit/len(train) * 100)}")

hit_test = 0
for row, desired_output in zip(*test):
    u_layer_1 = [sum(map(lambda x,y: x*y, inputs, neuron_w)) for neuron_w in weights]
    output = 0
    for u in u_layer_1:
        output = output | (u > 0)
    output = 1 if output else -1
    if output == desired_output[0]:
        hit_test += 1

print(f"Test: {hit_test} / {len(test)} = {int(hit_test/len(test) * 100)}")
print(f"TOTAL {hit+hit_test} / {(len(train)+len(test))} = {int((hit+hit_test)/(len(train)+len(test)) * 100)}")

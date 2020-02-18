# Many Adaptive Linear Neurons -  Rule 1   MR-I
from dataset import dataset_bipolar_output
from copy import copy
from math import exp
from functools import reduce

import random
#print("NOT WORKING")
#exit()

# Load data set
#train, test = dataset_bipolar_output('datasets/sonar.all-data')
train = [[1,1], [0,0], [1,0], [0,1]],  [[1],[1],[-1],[-1]]

inputs = train[0]
inputs = list(map(lambda x: x+[1], inputs))
ref_output = list(map(lambda x: x[0], train[1]))
layers = [20, 1]

training_vectors = list(zip(inputs, ref_output))
learning_coef = 0.4
theta = 0.5
weights = [[0 for _ in inputs[0]] for _ in range(layers[0])]

for epoch in range(20):
    print(epoch)
    for inputs, desired_output in training_vectors:
        u_layer_1 = [sum(map(lambda x,y: x*y, inputs, neuron_w)) for neuron_w in weights]
        output = 0
        for u in u_layer_1:
            output = output | (u > 0)
        output = 1 if output else -1
        error = (desired_output - output) * learning_coef 
        if not error: 
            continue

        adalines = enumerate(u_layer_1)
        adalines = filter(lambda x: abs(x[1]) < theta, adalines)
        if not adalines:
            continue
        #adalines = sorted(adalines, key=lambda x:x[1], reverse=desired_output == 1)
        #adalines[0] = 
        adalines = filter(lambda x: abs(x[1]) < theta, adalines)
        adalines = sorted(adalines, key=lambda x: abs(x[1]), reverse=True)

        for idx, u in adalines: 
            w_old = weights[idx]
            w_new = map(sum, zip(list(map(lambda x, e=error: x*e, inputs)), w_old))
            weights[idx] = list(w_new)
            _u_layer = [sum(map(lambda x,y: x*y, inputs, neuron_w)) for neuron_w in weights]
            output = 0
            for u in _u_layer:
                output = output | (u > 0)
            output = 1 if output else -1
            if output == desired_output:
                break
            weights[idx] = w_old
    

hit = 0
for row, desired_output in zip(*train):
    u_layer_1 = [sum(map(lambda x,y: x*y, inputs, neuron_w)) for neuron_w in weights]
    output = 0
    for u in u_layer_1:
        output = output | (u > 0)
    output = 1 if output else -1
    if output == desired_output[0]:
        hit += 1

print(f"Found {hit} / {len(train)} = {int(hit/len(train) * 100)}")
exit()
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

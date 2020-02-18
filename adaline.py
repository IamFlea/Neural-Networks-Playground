from dataset import dataset_bipolar_output
from copy import copy
import random
# Load data set
train, test = dataset_bipolar_output('datasets/sonar.all-data')

results = list(map(lambda x: x[0], train[1]))
x_table = [[1]+row for row in train[0]]
weights = [0 for _ in x_table[0]]

best_hit = 0
best_solution = copy(weights)
learning_coef = 0.001  # Use this bastard for learning!
for g in range(800):
    w_new_row = [weights]
    for x_row, desired_output in zip(x_table, results):
        output = sum(map(lambda x,y: x*y, x_row, weights))
        error = (desired_output - output) * learning_coef 
        w_new_row += [list(map(lambda x, e=error: x*e, x_row))]
    hit = 0
    for x_row, desired_output in zip(x_table, results):
        output = sum(map(lambda x,y: x*y, x_row, weights))
        output = 1 if output > 0 else -1 
        if output == desired_output:
            hit += 1
    #print(hit)
    weights = list(map(sum, zip(*w_new_row)))

print(f"Found {hit} / {len(train)} = {int(hit/len(train) * 100)}")
hit_test = 0
for row, output in zip(*test):
    sign = output[0]
    inputs = [1] + row
    result = sum(map(lambda x,y: x*y, inputs, weights)) 
    result = 1 if result > 0 else -1
    if result == sign:
        hit_test += 1

print(f"Test: {hit_test} / {len(test)} = {int(hit_test/len(test) * 100)}")
print(f"TOTAL {hit+hit_test} / {(len(train)+len(test))} = {int((hit+hit_test)/(len(train)+len(test)) * 100)}")

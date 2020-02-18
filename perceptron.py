from dataset import dataset_bipolar_output
from copy import copy
import random
# Load data set
train, test = dataset_bipolar_output('datasets/sonar.all-data')

z_table = []
w_table = []
# output = -+ 1
for row, output in zip(*train):
    sign = output[0]
    # Set weight vector to 0
    w_table += [[0 for w in row]]
    # Set `z vector` as a product of required `output * input` 
    z_table += [[sign]+[sign*col for col in row]]

w_row = [0 for w in train[0]]

best_hit = 0
best_solution = copy(w_row)

#print("z_table")
#print(z_table)
#print()
for g in range(800):
    hit = 0
    w_new_row = [copy(w_row)]


    for z_row in z_table:
        if sum(map(lambda x,y: x*y, w_row, z_row)) > 0:
            hit += 1
        else:
            w_new_row += [copy(z_row)]
    if hit > best_hit:
        best_hit = hit
        best_solution = copy(w_row)
    if len(w_new_row) == 1:
        print("FOUND")
        break
    #print('w_row', w_new_row)
    #print('Z',list(zip(*w_new_row)))
    w_row = list(map(sum, zip(*w_new_row)))

print(f"Found {best_hit} / {len(train)} = {int(best_hit/len(train) * 100)}")

hit = 0
for row, output in zip(*test):
    sign = output[0]
    inputs = [1] + row
    result = sum(map(lambda x,y: x*y, inputs, best_solution)) 
    result = 1 if result > 0 else -1
    if result == sign:
        hit += 1

print(f"Test: {hit} / {len(test)} = {int(hit/len(test) * 100)}")
print(f"TOTAL {hit+best_hit} / {(len(train)+len(test))} = {int((hit+best_hit)/(len(train)+len(test)) * 100)}")

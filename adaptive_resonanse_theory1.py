import numpy as np 
VIGILANCE = 0.6  # trashhold  0 - 1.0
LEARNING_COEF = 0.5  # standard 
train = np.array([[1,0,0,0,0,0],
                  [1,1,1,1,1,0],
                  [1,0,1,0,1,0],
                  [0,1,0,0,1,1],
                  [1,1,1,0,0,0],
                  [0,0,1,1,1,0],
                  [1,1,1,1,1,0],
                  [1,1,1,1,1,1]], np.float)

test = np.array([[1,1,1,1,1,1],
                 [1,1,1,1,1,0],
                 [1,1,1,1,0,0],
                 [1,1,1,0,0,0],
                 [1,1,0,0,0,0],
                 [1,0,0,0,0,0],
                 [0,0,0,0,0,0]], np.float)

L1_neurons_cnt = len(train[0])
L2_neurons_cnt = 1
# Init weights from the first neuron 
bottomUps = np.array([[1/(L1_neurons_cnt + 1) for _ in range(L1_neurons_cnt)]], np.float)
topDowns = np.array([[1 for _ in range(L1_neurons_cnt)]], np.float)

for tv in train: 
    print("-------")
    print('Train vector:', tv)
    createNewNeuron = True
    outputs = [bottomUps[i].dot(tv) for i in range(L2_neurons_cnt)]
    counter = L2_neurons_cnt
    while counter > 0:
        winning_output = max(outputs)
        winner_neuron_idx = outputs.index(winning_output)
        # NOTE!!! Sometimes there can be more than one winning neurons
        # Then we should select them randomly. For sake of simplicity,
        # this was not implemented for sake of simplicity 
        
        # Because `sum(tv)` can be 0 and we can not divide by zero :(
        tv_sum = sum(tv)
        if tv_sum == 0:
            similarity = 0 
        else:
            similarity = topDowns[winner_neuron_idx].dot(tv)/(sum(tv))
        print(" ", topDowns[winner_neuron_idx])
        print("    Bottom Ups Weights:", bottomUps[winner_neuron_idx])
        print("    Similartiy:", similarity)
        if similarity >= VIGILANCE:
            # Found similar neuron -> update their weights 
            createNewNeuron = False
            new_bottom_weights = tv * topDowns[winner_neuron_idx]/(LEARNING_COEF+tv.dot(topDowns[winner_neuron_idx]))
            new_top_weights = tv * topDowns[winner_neuron_idx]
            topDowns[winner_neuron_idx] = new_top_weights
            bottomUps[winner_neuron_idx] = new_bottom_weights
            break
        else:
            # Didn't find similar neuron
            outputs[winner_neuron_idx] = -1  # So it won't be selected in the next iteration
            counter -= 1
            
    if createNewNeuron:
        print("  Creating a new new neuron")
        new_bottom_weights = np.array([[i/(LEARNING_COEF + sum(tv)) for i in tv]], np.float)
        new_top_weights = np.array([[i for i in tv]], np.float)
        print("    Weights bottomUps:", new_bottom_weights)
        print("    Weights topDowns:", new_top_weights)
        bottomUps = np.append(bottomUps, new_bottom_weights, axis=0)
        topDowns = np.append(topDowns, new_top_weights, axis=0)
        L2_neurons_cnt += 1
    
print("=====")
print(f"Total Classes: {L2_neurons_cnt}")
print("Center of masses")
print(topDowns)
for tv in test:
    A = list(range(L2_neurons_cnt))
    createNewNeuron = True
    outputs = [bottomUps[i].dot(tv) for i in A]
    winning_weight = max(outputs)
    winner_neuron_idx = outputs.index(winning_weight)
    print(f"Class {winner_neuron_idx} for train vector {tv}")
     
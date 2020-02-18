import numpy as np 
from scipy.special import expit as sigmoid
from math import exp
# local files
import dataset 

class Backprop(object):
    EPOCH_SIZE = 10000
    LEARNING_COEF = 0.5
    HIDDEN_LAYERS = [5]
    
    def __init__(self, train, test=None, layers=HIDDEN_LAYERS, coef=LEARNING_COEF):
        super(Backprop, self).__init__()
        # Init training/testing set
        self.train_data = train
        self.test_data = test
        self.setDataset(train)
        # Init neural network
        self.layers = [len(self.inputs[0])]+layers+[len(self.outputs[0])]
        self.layers_cnt = len(self.layers)
        self.coef = coef
        self.error = 0 

        self.nn_biases = [np.zeros(i) for i in self.layers[1:]]
        self.nn_weights = [np.random.ranf((self.layers[idx+1], val)) for idx, val in enumerate(self.layers[:-1])]
        #self.nn_buff = [np.random.ranf((self.layers[idx+1], val)) for idx, val in enumerate(self.layers)]
        self.nn_buff = [np.zeros(val) for idx, val in enumerate(self.layers)]
        self.nn_derivation = [np.zeros(val) for idx, val in enumerate(self.layers)]

    def setDataset(self, dataset):
        self.inputs, self.outputs = dataset

    def sim(self, input):
        # Copies the pointer to input data, we do not edit nn_buff[0]
        self.nn_buff[0] = input
        # For each layer
        for idx, l_weights in enumerate(self.nn_weights):
            output_layer = self.nn_buff[idx+1]
            # Creates dot product to output_layer
            # And map sigmoid to output layer
            l_weights.dot(self.nn_buff[idx], out=output_layer) 
            np.add(output_layer, self.nn_biases[idx], out=output_layer)
            sigmoid(output_layer, out=output_layer)

    def train(self, input, output):
        self.sim(input)
        # Gets difference of these two
        np.subtract(self.nn_buff[-1], output, out=self.nn_derivation[-1])
        # Gets sum of squares
        self.error += np.sum(((self.nn_derivation[-1])**2))/2
        # Derivation for last layer
        np.multiply(self.nn_derivation[-1], -(self.nn_buff[-1]*(1-self.nn_buff[-1])), out=self.nn_derivation[-1])
        for idx in range(self.layers_cnt-2, 0, -1):
            np.dot(self.nn_weights[idx].transpose(), self.nn_derivation[idx+1], out=self.nn_derivation[idx])
            try:
                np.multiply(self.nn_derivation[idx], (1-self.nn_buff[idx])*self.nn_buff[idx], dtype=np.float, out=self.nn_derivation[idx])
            except:
                exit()
        for idx in range(self.layers_cnt-1, 0, -1):
            coef = self.coef*self.nn_derivation[idx]
            buff = self.nn_buff[idx-1]
            layer_weights = self.nn_weights[idx-1]
            for n_idx, neuron_weight in enumerate(layer_weights):
                np.add(neuron_weight, coef[n_idx]*buff, out=neuron_weight)
            np.add(self.nn_biases[idx-1], coef, out=self.nn_biases[idx-1])
        
    def run_sgd(self, epoch=EPOCH_SIZE):
        for x in range(epoch):
            self.error = 0
            for input, output in zip(self.inputs, self.outputs):
                #print("==Iteration==")
                self.train(input, output)
            print(x, self.error)

    def check(self):
        hit = 0
        for input, output in zip(self.inputs, self.outputs):
            self.sim(input)
            hit += int(output == (self.nn_buff[-1] >= 0.5))
        return hit, len(self.inputs)

nn = Backprop(*dataset.dataset_nary_output_np('datasets/sonar.all-data'), layers=[ ], coef=0.5)
#a, b = nn.test_data
#c, d = nn.train_data
#nn.setDataset((np.concatenate((a,c)), np.concatenate((b, d))))
nn.run_sgd(10000)
hit, total = nn.check()
print(hit, total, int(hit/total*100))
#exit()
nn.setDataset(nn.test_data)
hit, total = nn.check()
print(hit, total, int(hit/total*100))
exit(0)
for epoch in range(EPOCH_SIZE):
    print(epoch, nn.error)

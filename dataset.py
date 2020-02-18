import csv 
import random 
# lets be nice to those who wants to try `backpropagation.py` without numpy
try:
    import numpy as np 
except ModuleNotFoundError: 
    from sys import stderr 
    stderr.write("[WARNING] Numpy module was not found!\n")



TRAINING_PERCENTAGE = 0.8
classes = []
def _dataset_nary_output(filename, delimiter=',', quotechar='"'):
    global classes
    def _tryToConvert(row):
        global classes
        _class = row[-1]
        try:
            class_index = classes.index(_class)
        except ValueError:
            class_index = len(classes)
            classes += [_class]
        return list(map(float, row[:-1])) + [class_index]

    # Fills up data
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        allData = [_tryToConvert(row) for row in reader]   
    # Mew
    
    datasize = len(allData)
    result = random.sample(allData, datasize)
    #result = allData
    splitter = int(TRAINING_PERCENTAGE * datasize)
    return result[:splitter], result[splitter:]

class TV(list):
    def __len__(self):
        return len(self[0])

def dataset_nary_output(filename, output_len=1, delimiter=',', quotechar='"'):
    train, test = _dataset_nary_output(filename)
    train_inputs = [x[:-output_len] for x in train]
    train_outputs = [x[-output_len:] for x in train]
    test_inputs = [x[:-output_len] for x in test]
    test_outputs = [x[-output_len:] for x in test]
    train = TV([train_inputs, train_outputs])
    test = TV([test_inputs, test_outputs])
    return train, test

def dataset_nary_output_np(filename, output_len=1, delimiter=',', quotechar='"'):
    train, test = dataset_nary_output(filename, output_len, delimiter, quotechar)
    transfer = lambda x: np.array(x, dtype=np.float)
    train = transfer(train[0]), transfer(train[1])
    test = transfer(test[0]), transfer(test[1])
    return train, test

    


def dataset_bipolar_output(filename, delimiter=',', quotechar='"'):
    train, test = dataset_nary_output(filename)
    return train, test


if __name__ == '__main__':
    train, test = dataset_nary_output_np('datasets/sonar.all-data')
    print(train[1][0])
    pass

    #print(float("0.2"))



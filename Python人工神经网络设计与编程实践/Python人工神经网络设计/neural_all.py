import numpy

import scipy.special

import matplotlib.pyplot
'''
%matplotlib inline
'''


class neuralNetwork:


    def __init__(self,inputnodes,hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0, pow(self.inodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.onodes,self.hnodes))

        #learning rate
        self.lr = learningrate

        #
        self.activation_function = lambda x:scipy.special.expit(x)

        pass

    #train the neural network
    def train(self, inputs_list, targets_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #
        hidden_inputs = numpy.dot(self.wih, inputs)
        #
        hidden_outputs = self.activation_function(hidden_inputs)

        #
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #
        final_outputs = self.activation_function(final_inputs)

        #
        output_errors = targets - final_outputs
        #
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        #
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    #query the neural network
    def query(self, inputs_list):
        #
        inputs = numpy.array(inputs_list, ndmin=2).T

        #
        hidden_inputs = numpy.dot(self.wih, inputs)
        #
        hidden_outputs = self.activation_function(hidden_inputs)

        #
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

#number of input, hidden and ouput nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

#learning rate
learning_rate = 0.1

#
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


#load the mnist training data csv file into a list
training_data_file = open("MNIST-csv-all/mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 5

for e in range(epochs):
    #go through all records in the training data set
    for record in training_data_list:
        #
        all_values = record.split(',')
        #
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        #
        targets = numpy.zeros(output_nodes) + 0.01
        #
        targets[int(all_values[0])] = 0.99
        n.train(inputs,  targets)
        pass
    pass

#7
#load the mnist test data csv file into a list
test_data_file = open("MNIST-csv-all/mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()


#8
#scorecard for how well he network performs, initially empty
scorecard = []

#go through all the records in the test data set
for record in test_data_list:
    #split the record by the ',' commas
    all_values = record.split(',')
    #
    correct_label = int(all_values[0])
    print(correct_label,"correct_label")
    #
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #
    outputs = n.query(inputs)
    #
    label = numpy.argmax(outputs)
    print(label,"network's answer")
    #
    if (label == correct_label):
        #
        scorecard.append(1)
    else:
        #
        scorecard.append(0)
        pass
    pass


#calculate the performance score, the fraction of correct answers
#scorecard_array = numpy.asarray(scorecard)#???
scorecard_array = numpy.asfarray(scorecard)
print("nerutal network test for all \
        performance = ", scorecard_array.sum() / scorecard_array.size)

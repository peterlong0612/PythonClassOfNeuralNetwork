##################实验3.2
import numpy
# scipy
import scipy.special
import scipy.ndimage
import matplotlib.pyplot
# %matplot inline

# neural network class definition
class neuralNetwork:

    # init
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link
        # weights
        # w11
        # w12
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning
        self.lr = learningrate

        # activation
        self.activation_function = lambda x: scipy.special.expit(x)

        pass


    #train the neural network
    def train(self, inputs_list, targets_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weight for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    #query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# number
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.01

#create
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#train

training_data_file = open("MNIST-csv-all/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train

# epochs 
epochs = 10

for e in range(epochs):
    # go through
    for record in training_data_list:
        # split
        all_values = record.split(',')
        # scale
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 *0.99) +0.01
        # create
        targets = numpy.zeros(output_nodes) +0.01
        # all_values[0]
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        ##
        # rotate
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        #
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)
        
        pass
    pass

#test
test_data_file = open("MNIST-csv-all/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test

# scorecard
scorecard = []

# go through all
for record in test_data_list:
    # split
    all_values = record.split(',')
    # correct answer
    correct_label = int(all_values[0])
    # scale and shift
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) +0.01
    # query
    outputs = n.query(inputs)
    # the index
    label = numpy.argmax(outputs)
    # append
    if(label == correct_label):
        # network's answer
        scorecard.append(1)
    else:
        # network's answer doesn't
        scorecard.append(0)
        pass

    pass

scorecard_array = numpy.asfarray(scorecard)
print(" dataset------all \
             performance = ", scorecard_array.sum() / scorecard_array.size )
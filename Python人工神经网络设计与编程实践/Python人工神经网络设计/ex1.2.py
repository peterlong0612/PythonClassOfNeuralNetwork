#######################################1.2
import numpy
# scipy
import scipy.special
# library
import matplotlib.pyplot
# ensure
#%matplotlib inline

#helper to
import imageio

#neural network class definition
class neuralNetwork:

    #init
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
learning_rate = 0.1

#create
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#train
training_data_file = open("MNIST-csv-mini/train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train
epochs = 10

for e in range(epochs):
    # go through
    for record in training_data_list:
        # split
        all_values = record.split(',')
        # scale
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) +0.01
        # create
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0]
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# test the neural network with our own images

# load
print("loading ... my image")
img_array = imageio.imread('3.png', as_gray = True)

# reshape
img_data = 255.0 - img_array.reshape(784)

# then scale
img_data = (img_data / 255.0 * 0.99) + 0.01
print("min = ", numpy.min(img_data))
print("max = ", numpy.max(img_data))

# plot
matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

# query
outputs = n.query(img_data)
print("outputs", outputs)

# the index of the highest
label = numpy.argmax(outputs)
print("network says ", label)

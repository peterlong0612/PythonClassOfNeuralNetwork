##################实验3.1
import numpy
import matplotlib.pyplot
#%matplot inline

# scipy
import scipy.ndimage
# open
data_file = open("MNIST-csv-all/mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# which
record = 6

# scale
all_values = data_list[record].split(',')
scaled_input = ((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01).reshape(28,28)

print(numpy.min(scaled_input))
print(numpy.max(scaled_input))

# plot
matplotlib.pyplot.imshow(scaled_input, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
# create
# rotated
inputs_plus10_img = scipy.ndimage.rotate(scaled_input, 10.0, cval=0.01, order=1, reshape=False)
# rotate clockwise
inputs_minus10_img = scipy.ndimage.rotate(scaled_input, -10.0, cval=0.01, order=1, reshape=False)

print(numpy.min(inputs_plus10_img))
print(numpy.max(inputs_plus10_img))

# plot the +10
matplotlib.pyplot.imshow(inputs_plus10_img, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
# plot the -10
matplotlib.pyplot.imshow(inputs_minus10_img, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

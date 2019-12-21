###########实验一
# helper to load data from PNG image files
import imageio
# 
import glob

import numpy
#
import matplotlib.pyplot
# 
# %matplotlib online

# our own image tst data set
our_own_dataset = []

for image_file_name in glob.glob('4.png'):
    print("loading ... ",image_file_name)
    #
    label = int(image_file_name[0])
    # load image data from png files into an array
    img_array = imageio.imread(image_file_name, as_gray=True)
    # reshape
    img_data = 255.0 - img_array.reshape(784)
    # then scale data to range from 0.01-01.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    # append
    record = numpy.append(label, img_data)
    print(record)
    our_own_dataset.append(record)
    pass

#matplotlib.pyplot.imshow(our_own_dataset[3][1:].reshape(28,28), cmap='Greys', interpolation = 'None')
matplotlib.pyplot.imshow(our_own_dataset[0][1:].reshape(28, 28), cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

print(our_own_dataset[0])

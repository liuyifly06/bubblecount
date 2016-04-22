import sys, traceback, time, skflow
import os.path
import numpy as np
import tensorflow as tf
import dataSet as ds
from matplotlib import pyplot as plt
from skimage import io
from .. import GlobalVariables as gv
from ..PreProcess.readinfo import getInfo

def train(batchNum = 500, batchSize = 200000, learningRate = 0.001,
          ImagePatchWidth = 20, #layers = [500, 1000, 500],
          ImagePatchStep = 4, labelOptionNum = 100,
          labelMode = 'NUM'):
    trainDS = ds.read_data_sets(ImagePatchWidth, ImagePatchStep,
                                labelOptionNum, 'train', labelMode)

    classifier = skflow.TensorFlowEstimator(
        model_fn = conv_model,
        n_classes = labelOptionNum,
        batch_size = batchSize,
        steps = batchNum,
        learning_rate = learningRate)

    classifier.fit(trainDS.images, np.argmax(trainDS.labels, axis = 1),
                   logdir = gv.__DIR__ + gv.tensorflow_log_dir)
    return classifier


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

def conv_model(image, y):
    # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and height
    # final dimension being the number of color channels
    image = tf.reshape(image, 
                       [-1, image.shape[0], image.shape[1], image.shape[3]])
    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(image, n_filters=32, filter_shape=[5, 5], 
                                    bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5], 
                                    bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # densely connected layer with 1024 neurons
    h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, keep_prob=0.5)
    return skflow.models.logistic_regression(h_fc1, y)


def test(classifier, ImagePatchWidth = 20, ImagePatchStep = 4,
         labelOptionNum = 100, labelMode = 'PRO'):
    image_files, bubble_num, bubble_regions = getInfo()

    result_filename   = gv.cnn__result_filename
    accuracy_filename = gv.cnn__accuracy_filename
    
    result   = np.zeros((len(image_files),1))
    accuracy = np.zeros((len(image_files),4))

    index = -1
    start_time = time.time()

    PROGRESS = progress.progress(0, len(image_files), prefix_info = 'Labeling ')

    for i, image_file in enumerate(image_files):
        testDS = ds.read_data_sets(ImagePatchWidth, ImagePatchStep,
                                   labelOptionNum, 'test', labelMode,
                                   imageName = image_file)
        y = classifier.predict(testDS.images)
        index = index + 1
        result[index] = np.sum(y)      
        # saving labeled result as image
        io.imsave(gv.__DIR__ + gv.cnn__image_dir + image_file,
                  np.reshape(y, (testDS.ylength, testDS.xlength)))
        _y = np.argmax(testDS.labels, axis = 1)
        # total accuracy
        accuracy[index, 0] = np.true_divide(np.sum(y == _y), _y.size)
        # accuracy of negative labeled instances
        accuracy[index, 1] = np.true_divide(np.sum(np.all(
            [y == _y, _y == 0], axis = 0)), np.sum(_y == 0))
        # accuracy of positive labeled instances
        accuracy[index, 2] = np.true_divide(np.sum(np.all(
            [y == _y, _y >  0], axis = 0)), np.sum(_y >  0))
        # average difference sum
        accuracy[index, 3] = np.true_divide(
            np.sum(np.absolute(np.subtract(y, _y))), _y.size)

        PROGRESS.setCurrentIteration(i+1)
        PROGRESS.setInfo(suffix_info = image_file)
        PROGRESS.printProgress()
    accuracy.tofile(accuracy_filename, sep =" ")
    return [result, accuracy]
    
def main():
    try:      
        image_files, bubble_num, bubble_regions = getInfo()
        if not os.path.isfile(gv.cnn__result_filename):
            ins_size = 100
            stride = 10
            label_option = 100
            #training data
            classifier = train(batchNum = 10000,
                               batchSize = 2000,
                               learningRate = 0.01,
                               ImagePatchWidth=ins_size,
                               ImagePatchStep = stride,
                               labelOptionNum = label_option,
                               labelMode = 'PRO')
            result, accuracy = test(classifier, ins_size, stride, label_option)
        result = np.loadtxt(gv.cnn__result_filename)
        accuracy = np.loadtxt(gv.cnn__accuracy_filename)
        accuracy = np.reshape(accuracy, (len(accuracy)/4,4))
        fig, ax = plt.subplots(1,2)
        ax[0].scatter(bubble_num,result)
        ax[1].plot(accuracy)
        plt.show()
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

#cnn__image_dir = '/images/convolutionalNN/'
#cnn__result_filename = 'convolutionalNNResult.txt'
#cnn__accuracy_filename = 'convolutionalNNAccuracy.txt'

if __name__ == '__main__':
    main()

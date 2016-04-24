# Deeplearning neural network with  skflow
# Skflow use google machine learning tool TensorFlow

import sys, traceback, time, skflow
import os.path
import numpy as np
import tensorflow as tf
import dataSet as ds
from matplotlib import pyplot as plt
from skimage import io
from .. import GlobalVariables as gv
from ..PreProcess.readinfo import getInfo
from ..ProgressBar import progress

def train(batchNum = 500, batchSize = 200000, learningRate = 0.001,
          layers = [500, 1000, 500], ImagePatchWidth = 20,
          ImagePatchStep = 4, labelOptionNum = 100,
          labelMode = 'PRO'):
    trainDS = ds.read_data_sets(ImagePatchWidth, ImagePatchStep,
                                labelOptionNum, 'train', labelMode);
    print ('Training deep learning neural network ...')
    classifier = skflow.TensorFlowDNNClassifier(
        hidden_units = layers,
        n_classes = labelOptionNum,
        batch_size = batchSize,
        steps = batchNum,
        learning_rate = learningRate)
    classifier.fit(trainDS.images, np.argmax(trainDS.labels, axis = 1),
                   logdir = gv.__DIR__ + gv.tensorflow_log_dir)
    return classifier

def test(classifier, ImagePatchWidth = 20, ImagePatchStep = 4,
         labelOptionNum = 100, labelMode = 'PRO'):
    image_files, bubble_num, bubble_regions = getInfo()

    result_filename   = gv.dp__result_filename
    accuracy_filename = gv.dp__accuracy_filename
    
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
        io.imsave(gv.__DIR__ + gv.dp__image_dir + image_file,
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
    accuracy.tofile(accuracy_filename, sep = " ")
    result.tofile(result_filename, sep = " ")
    return [result, accuracy]
    
def main():
    try:      
        image_files, bubble_num, bubble_regions = getInfo()
        if not os.path.isfile(gv.dp__result_filename):
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
                               labelMode = 'NUM')
            result, accuracy = test(classifier, ins_size, stride, label_option)
        result = np.loadtxt(gv.dp__result_filename)
        accuracy = np.loadtxt(gv.dp__accuracy_filename)
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

if __name__ == '__main__':
    main()

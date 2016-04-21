# Deeplearning neural network with  skflow
# Skflow use google machine learning tool TensorFlow

import sys, traceback, time, random, skflow
import os.path
import numpy as np
import tensorflow as tf
import dataSet as ds
from matplotlib import pyplot as plt
from skimage import io, img_as_float
from sklearn import datasets, cross_validation, metrics
from .. import GlobalVariables as gv
from ..PreProcess.readinfo import getInfo

def displayTime(str_process ,elap_time, rema_time):
    if(rema_time < 0):
        print    'Current process : ' + str_process + \
                     '|| Time Elapsed : ' + \
            time.strftime(" %H:%M:%S", time.gmtime(elap_time))
    else:
        print    'Current process : ' + str_process + \
                     '|| Time Elapsed : ' + \
            time.strftime(" %H:%M:%S", time.gmtime(elap_time)) + \
                      '|| Time Remaining : ' + \
            time.strftime(" %H:%M:%S", time.gmtime(rema_time))
def train(_steps = 500, _batch_size = 200000, _learning_rate = 0.001, \
          _layers = [500, 1000, 500], _ins_size = 20, _step = 4, \
          _label_option = 100):
    random.seed(42)
    trainDS = ds.read_data_sets(_ins_size, _step, _label_option, 'train');
    classifier = skflow.TensorFlowDNNClassifier(
        hidden_units = _layers,
        n_classes = _label_option,
        batch_size = _batch_size,
        steps = _steps,
        learning_rate = _learning_rate)
    classifier.fit(trainDS.images, np.argmax(trainDS.labels, axis = 1))
    return classifier

def test(classifier, _ins_size = 20, _step = 4, _label_option = 100):
    image_files, bubble_num, bubble_regions = getInfo()

    result_filename   = gv.dp__result_filename
    accuracy_filename = gv.dp__accuracy_filename
    
    result   = np.zeros((len(image_files),1))
    accuracy = np.zeros((len(image_files),4))

    index = -1
    start_time = time.time()

    for image_file in image_files:
        testDS = ds.read_data_sets(_ins_size, _step, _label_option, \
                                     'test', imageName = image_file)
        y = classifier.predict(testDS.images)
        index = index + 1
        result[index] = np.sum(y)
        
        # saving labeled result as image
        io.imsave(gv.__DIR__ + gv.dp__image_dir + image_file,\
            np.reshape(y, (testDS.ylength, testDS.xlength)))
        # total accuracy
        accuracy[index, 0] = np.sum(y == np.argmax(testDS.labels, axis = 1)) \
           / testDS.labels.size
        # accuracy of negative labeled instances
        accuracy[index, 1] = np.sum(np.all( \
           [y == np.argmax(testDS.labels, axis = 1), \
            np.argmax(testDS.labels, axis = 1) == 0], \
           axis = 0) ) / np.sum(testDS.labels == 0)
        # accuracy of positive labeled instances
        accuracy[index, 2] = np.sum(np.all( \
           [y == np.argmax(testDS.labels, axis = 1), \
            np.argmax(testDS.labels, axis = 1) >  0], \
           axis = 0) ) / np.sum(testDS.labels >  0)
        # average difference sum
        accuracy[index, 3] = np.sum(np.absolute(np.subtract(y, \
           np.argmax(testDS.labels, axis = 1)))) / testDS.labels.size
        # print time information
        elapse_time = (time.time()-start_time)
        if(elapse_time >= 1):
            remain_time = elapse_time / (index+1) * len(image_files) \
                          - elapse_time
            print 'Processing .. ' + image_file + \
                  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) + \
                  ' has past. ' + 'Remaining time: ' + \
                  time.strftime(" %H:%M:%S", time.gmtime(remain_time))
        else:
            print 'Processing .. ' + image_file + \
                  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) + \
                  ' has past'
    result.tofile(result_filename, sep=" ")
    accuracy.tofile(accuracy_filename, sep =" ")
    return [result, accuracy]

def main():
    try:      
        image_files, bubble_num, bubble_regions = getInfo()
        if not os.path.isfile(gv.dp__result_filename):
            ins_size = 20
            step = 4
            label_option = 100
            #training data
            classifier = train(_steps = 10000, _batch_size = 20000, \
                               _learning_rate = 0.01, \
                               _ins_size = ins_size, _step = step, \
                               _label_option = label_option)
            result, accuracy = test(classifier, ins_size, step, label_option)
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

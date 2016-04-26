# Deeplearning neural network with  skflow
# Skflow use google machine learning tool TensorFlow

## for server##
#import matplotlib
#matplotlib.use('Agg')
###############

import sys, traceback, time, skflow
import os.path
import numpy as np
import dataSet as ds
import tensorflow as tf
from skimage import io
from scipy.stats import linregress
from matplotlib import pyplot as plt
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

    if(gv.log_write):
        classifier.fit(trainDS.images, np.argmax(trainDS.labels, axis = 1),
                   logdir = gv.__DIR__ + gv.tensorflow_log_dir)
    else:
        classifier.fit(trainDS.images, np.argmax(trainDS.labels, axis = 1))
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
        if(gv.dp_image_save):
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
    if(gv.dp_test_save_data):
        accuracy.tofile(accuracy_filename, sep = " ")
        result.tofile(result_filename, sep = " ")
    return [result, accuracy]

def tuningParameters( batch_num = 10000,
                      batch_size = 2000,
                      learning_rate = [0.001, 0.005, 0.01, 0.05],
                      ins_size = [10, 20, 50, 100, 150],
                      stride = [5],
                      label_option = [10, 100, 1000],
                      label_mode =['PRO', 'NUM'] ):
    image_files, bubble_num, bubble_regions = getInfo()
    par_evaluation = np.array([])
    total = (len(learning_rate) * len(ins_size) * len(stride) *
             len(label_option) * len(label_mode))
    iteration = 0
    PROGRESS = progress.progress(iteration, total,
            prefix_info = 'Tuning Parameters')
    for ins in ins_size:
        for s in stride:
            for lo in label_option:
                for lr in learning_rate:
                    for lm in label_mode:
                        info = ('ins:'+str(ins)+' s:'+str(s)+' classes:'+str(lo)
                            +' LR:' +str(lr)+' mode:'+lm)
                        
                        iteration = iteration + 1
                        current_row = performance(ins, s, lo, batch_num,
                                                  batch_size, lr, lm)

                        if(par_evaluation.size == 0):
                            par_evaluation = current_row
                        else:
                            np.append(par_evaluation, current_row, axis = 0)

                        PROGRESS.setInfo(suffix_info = info)
                        PROGRESS.setCurrentIteration(iteration)
                        PROGRESS.printProgress()
                        directory = gv.__DIR__ + gv.dp__tuningPar_dir
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        par_evaluation.tofile(directory
                            + gv.dp__tuningPar_filename, sep = ' ')


def test_run(ins_size = 100, stride = 10, label_option = 100, batch_num = 10000,
        batch_size = 2000, learning_rate = 0.01, label_mode = 'NUM',
        run_mode = 'NO'):
    
    image_files, bubble_num, bubble_regions = getInfo()
    if (not os.path.isfile(gv.dp__result_filename)) or (run_mode == 'YES'):
        #training data
        classifier = train(batchNum = batch_num,
                           batchSize = batch_size,
                           learningRate = learning_rate,
                           ImagePatchWidth=ins_size,
                           ImagePatchStep = stride,
                           labelOptionNum = label_option,
                           labelMode = label_mode)
        result, accuracy = test(classifier, ins_size, stride, label_option)
    result = np.loadtxt(gv.dp__result_filename)
    accuracy = np.loadtxt(gv.dp__accuracy_filename)
    accuracy = np.reshape(accuracy, (len(accuracy)/4,4))
    slope, intercept, r_value, p_value, std_err = (linregress(bubble_num,
                                                         result.T))    
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(bubble_num, result)
    ax[0].plot([min(bubble_num), max(bubble_num)],
               [min(bubble_num)*slope + intercept,
                max(bubble_num)*slope + intercept], c = 'r')
    ax[1].plot(accuracy)
    plt.show()

def performance(ins_size = 100, stride = 10, label_option = 100,
                batch_num = 10000, batch_size = 2000, learning_rate = 0.01,
                label_mode = 'NUM'):
    image_files, bubble_num, bubble_regions = getInfo()
    classifier = train(batchNum = batch_num,
                       batchSize = batch_size,
                       learningRate = learning_rate,
                       ImagePatchWidth=ins_size,
                       ImagePatchStep = stride,
                       labelOptionNum = label_option,
                       labelMode = label_mode)
    result, accuracy = test(classifier, ins_size, stride, label_option)
    slope, intercept, r_value, p_value, std_err = linregress(bubble_num,
                                                         result.T)

    accuracy_mean = np.mean(accuracy, axis = 0)
    accuracy_std  = np.std (accuracy, axis = 0)
    status        = np.append(np.array([slope, intercept, r_value, 
                                        p_value, std_err]),
                              np.append(accuracy_mean, accuracy_std))

    # save result
    directory = gv.__DIR__ + gv.dp__tuningPar_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    result.tofile(directory + 'patch_' + str(ins_size) + '_' + str(stride)
                  + '_label_' + str(label_option) + '_' + str(label_mode)
                  + '_training_' + str(learning_rate) + '_' + str(batch_num)
                  + '_' + str(batch_size) + '_result.dat')
    accuracy.tofile(directory + 'patch_' + str(ins_size) + '_' + str(stride)
                  + '_label_' + str(label_option) + '_' + str(label_mode)
                  + '_training_' + str(learning_rate) + '_' + str(batch_num)
                  + '_' + str(batch_size) + '_accuracy.dat')
    
    return status 
                  
def main():
    try:      
        if len(sys.argv) > 1:
            print('Runing ')
            performance(ins_size = 20,
                stride = 20,
                label_option = 100,
                batch_num = 1,
                batch_size = 200,
                learning_rate = 0.01,
                label_mode = 'PRO') #,
                #run_mode = sys.argv[1])
        else:
            print ('Parameter Tuning')
            tuningParameters(batch_num = 10000,
                      batch_size = 2000,
                      learning_rate = [0.001, 0.01, 0.05],
                      ins_size = [20, 50, 100, 200],
                      stride = [10, 20],
                      label_option = [100, 1000],
                      label_mode =['PRO', 'NUM'] )
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

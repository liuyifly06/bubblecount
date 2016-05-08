# Deeplearning neural network with  skflow
# Skflow use google machine learning tool TensorFlow

## for server##
#import matplotlib
#matplotlib.use('Agg')
###############

import time, skflow, multiprocessing, os
import os.path
import numpy as np
import dataset as ds
import tensorflow as tf
from skimage import io
from scipy.stats import linregress
from matplotlib import pyplot as plt
from bubblecount import globalvar as gv
from bubblecount.preprocess.readinfo import getinfo
from bubblecount.progressbar import progress

def train(batchNum = 500, batchSize = 200, learningRate = 0.001,
          layers = [200, 400, 200], ImagePatchWidth = 20,
          ImagePatchStep = 4, labelOptionNum = 0,
          labelMode = 'PRO'):

    trainDS = ds.read_data_sets(ImagePatchWidth, ImagePatchStep,
                                labelOptionNum, 'train', labelMode);
    print ('Training deep learning neural network ...')
    """TensorFlow-skflow DNN Classifier model.
    Parameters:
      hidden_units: List of hidden units per layer.
      n_classes: Number of classes in the target.
      batch_size: Mini batch size.
      steps: Number of steps to run over data.
      optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
      learning_rate: If this is constant float value, no decay function is used.
        Instead, a customized decay function can be passed that accepts
        global_step as parameter and returns a Tensor.
        e.g. exponential decay function:
        def exp_decay(global_step):
            return tf.train.exponential_decay(
                learning_rate=0.1, global_step,
                decay_steps=2, decay_rate=0.001)
      class_weight: None or list of n_classes floats. Weight associated with
        classes for loss computation. If not given, all classes are
        supposed to have weight one.
      continue_training: when continue_training is True, once initialized
        model will be continuely trained on every call of fit.
      config: RunConfig object that controls the configurations of the
        session, e.g. num_cores, gpu_memory_fraction, etc.
      dropout: When not None, the probability we will drop out a given coordinate.
    """
    classifier = skflow.TensorFlowDNNClassifier(
        hidden_units      = layers,
        n_classes         = labelOptionNum,
        batch_size        = batchSize,
        steps             = 1,
        optimizer         = 'Adagrad', #"SGD", "Adam", "Adagrad"
        learning_rate     = learningRate,
        continue_training = True)
    """
        class_weight      = None,
        clip_gradients    = 5.0,
        config            = None,
        verbose           = 1,
        dropout           = None 
    """
    probar = progress.progress(0, batchNum)
    gv.log_write = False
    for i in range(batchNum):
        probar.setCurrentIteration(i+1)
        probar.setInfo(prefix_info = 'Training ...',
                       suffix_info = 'Batch: ' + str(i+1) + '/' + str(batchNum))
        probar.printProgress()
        images, labels = trainDS.next_batch(batchSize)
        if(gv.log_write):
            classifier.fit(images, labels,
                           logdir = gv.__DIR__ + gv.tensorflow_log_dir)
        else:
            classifier.fit(images, labels)
    return classifier

def test(classifier, ImagePatchWidth = 20, ImagePatchStep = 4,
         labelOptionNum = 0, labelMode = 'PRO'):
    image_files, bubble_num, bubble_regions = getinfo()

    result_filename   = gv.dp__result_filename
    accuracy_filename = gv.dp__accuracy_filename
    
    result   = np.zeros((len(image_files),1))
    correct  = np.zeros(4)
    accuracy = np.zeros(4)
    totalnum_instances = 0
    index = -1
    PROGRESS = progress.progress(0, len(image_files), prefix_info = 'Labeling ')
    
    labeled_number = np.zeros((len(image_files),1))

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
        labeled_number[index] = np.sum(_y)
        current_correct =  np.array([np.sum(y == _y),
                           np.sum(np.all([y == _y, _y == 0], axis = 0)),
                           np.sum(np.all([y == _y, _y >  0], axis = 0)),
                           np.sum(np.absolute(np.subtract(y, _y)))])
        correct = np.add(correct, current_correct)
        totalnum_instances =  _y.size + totalnum_instances
      
        PROGRESS.setCurrentIteration(i+1)
        PROGRESS.setInfo(suffix_info = image_file)
        PROGRESS.printProgress()
    
    accuracy = np.true_divide(correct, totalnum_instances)
    if(gv.dp_test_save_data):
        accuracy.tofile(accuracy_filename, sep = " ")
        result.tofile(result_filename, sep = " ")
        labeled_number.tofile('labeled_number.dat', sep = " ")
    return [result, accuracy]

def multi_run_wrapper(args):
    return performance(*args)

def tuningParameters( MaxProcessNum = 8,
                      batch_num = [10000],
                      batch_size = [2000],
                      learning_rate = [0.001, 0.005, 0.01, 0.05],
                      ins_size = [10, 20, 50, 100, 150],
                      stride = [5],
                      label_option = [10, 100, 1000],
                      label_mode =['PRO', 'NUM'] ):
    
    gv.dp_test_save_data = False
    gv.dp_image_save = False
    gv.log_write = False
    gv.show_progress = False
    # get all possible parameters
    info  = []
    pars  = []
 
    iteration = -1
    for bn in batch_num:
      for bs in batch_size:
        for ins in ins_size:
          for s in stride:
            for lo in label_option:
              for lr in learning_rate:
                for lm in label_mode:
                  iteration = iteration + 1
                  info.append(str(bn) + ' ' + str(bs) +  ' ' + str(ins)+ ' ' +
                              str(s)  + ' ' + str(lo) + ' ' + str(lr) + ' ' +
                              lm)
                  pars.append((ins, s, lo, bn, bs, lr, lm))

    # calculating with all the parameters
    if(MaxProcessNum <= 0):
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(processes = MaxProcessNum)
    evaluation = pool.map(multi_run_wrapper, pars)
    pool.close()
    pool.join()
    
    FILE_HEAD = ['batch_num', 'batch_size', 'ins_size', 'stride',
                 'label_option', 'learning_rate', 'label_mode',
                 'slope', 'intercept', 'r_value', 'p_value', 'std_err'
                 'accuracy_total', 'accuracy_neg', 'accuracy_pos',
                 'accuracy_distance']

    directory = gv.__DIR__ + gv.dp__tuningPar_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(directory + gv.dp__tuningPar_filename, "w")
    for item in FILE_HEAD:
        f.write(item + ' ')
    f.write('\n')
    for info_line, eval_res in zip(info, evaluation):
        f.write(info_line)
        for item in eval_res:
            f.write(' %f' % item)
        f.write('\n')  
    f.close()
    return [info, evaluation]

def test_run(ins_size = 100, stride = 10, label_option = 0, batch_num = 10000,
        batch_size = 2000, learning_rate = 0.01, label_mode = 'NUM',
        run_mode = 'NO'):
    
    image_files, bubble_num, bubble_regions = getinfo()
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
    print 'slope intercept r_value p_value std_err'
    print slope, intercept, r_value, p_value, std_err
    plt.show()

def performance(ins_size = 100, stride = 10, label_option = 100,
                batch_num = 10000, batch_size = 2000, learning_rate = 0.01,
                label_mode = 'NUM'):
    image_files, bubble_num, bubble_regions = getinfo()
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
    status        = np.append(np.array([slope, intercept, r_value, 
                                        p_value, std_err]), accuracy)
    # save result
    directory = gv.__DIR__ + gv.dp__tuningPar_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    result.tofile(directory + 'patch_' + str(ins_size) + '_' + str(stride)
                  + '_label_' + str(label_option) + '_' + str(label_mode)
                  + '_training_' + str(learning_rate) + '_' + str(batch_num)
                  + '_' + str(batch_size) + '_result.dat')    
    return status 

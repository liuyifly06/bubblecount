# Deeplearning neural network with  skflow
# Skflow use google machine learning tool TensorFlow

## for server##
#import matplotlib
#matplotlib.use('Agg')
###############

import sys, time, multiprocessing, os
import os.path
import numpy as np
import dataset as ds
import skimage.io as io
import tensorflow.contrib.learn as skflow
import tensorflow as tf
from scipy.stats import linregress
from matplotlib import pyplot as plt
from bubblecount import globalvar as gv
from bubblecount.preprocess.readinfo import getinfo
from bubblecount.progressbar import progress

def train(batchNum = 500,
          batchSize = 200,
          ImagePatchWidth = 20,
          ImagePatchStep = 4,
          labelMode = 'PRO',
          label_mutiplier = 1.0,
          hidden_units = [200, 400, 200],
          steps = 200,
          optimizer = 'Adagrad', #"SGD", "Adam", "Adagrad"
          learning_rate = 0.001,
          clip_gradients = 5.0,
          config = None,
          verbose = 1,
          dropout = None):
    """Train deep neural-network.
    Parameters:
      hidden_units: List of hidden units per layer.
      batch_size: Mini batch size.
      steps: Number of steps to run over data.
      optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
      learning_rate: If this is constant float value, no decay function is
        used. Instead, a customized decay function can be passed that accepts
        global_step as parameter and returns a Tensor.
        e.g. exponential decay function:
        def exp_decay(global_step):
            return tf.train.exponential_decay(
                learning_rate=0.1, global_step,
                decay_steps=2, decay_rate=0.001)
      continue_training: when continue_training is True, once initialized
        model will be continuely trained on every call of fit.
      config: RunConfig object that controls the configurations of the session,
        e.g. num_cores, gpu_memory_fraction, etc.
      verbose: Controls the verbosity, possible values:
        0: the algorithm and debug information is muted.
        1: trainer prints the progress.
        2: log device placement is printed.
      dropout: When not None, the probability we will drop out a given coordinate.
    """
    print ('Training deep learning neural network ...')
    # generate data set
    trainDataset = ds.read_data_sets(
        instanceSize = ImagePatchWidth,
        stride = ImagePatchStep,
        instanceMode = 'train',
        labelMode = labelMode,
        label_mutiplier = label_mutiplier)
   
    # deep neural-network regression class (DNNRegressor)
    classifier = skflow.TensorFlowDNNRegressor(
        hidden_units = hidden_units,
        batch_size = 32,
        steps = steps,
        optimizer = optimizer, #"SGD", "Adam", "Adagrad"
        learning_rate = learning_rate,
        continue_training = True,
        clip_gradients = clip_gradients,
        config = config,
        verbose = verbose,    
        dropout = dropout)
   
    # train the DNNRegressor on generated data set
    probar = progress.progress(0, batchNum)
    gv.log_write = False
    for i in range(batchNum):
        probar.setCurrentIteration(i+1)
        probar.setInfo(
            prefix_info = 'Training ...',
            suffix_info = 'Batch: ' + str(i+1) + '/' + str(batchNum))
        probar.printProgress()
        images, labels = trainDataset.next_batch(batchSize)
        if(gv.log_write):
            classifier.fit(images, labels,
                           logdir = gv.__DIR__ + gv.tensorflow_log_dir)
        else:
            classifier.fit(images, labels)
    return classifier

def test(classifier,
         filename,
         ImagePatchWidth = 20,
         ImagePatchStep = 4,
         labelMode = 'PRO',
         label_mutiplier = 1.0,
         plot_show = 1,
         save_image = True):
    """ label image with trained deep neural-network
    """
    if(labelMode == 'NUM' and ImagePatchStep < ImagePatchWidth):
        ImagePatchStep = ImagePatchWidth
    
    # generate test data
    testDS = ds.read_data_sets(
        instanceSize = ImagePatchWidth,
        stride = ImagePatchStep,
        instanceMode = 'test',
        labelMode = labelMode,
        imageName = filename,
        label_mutiplier = label_mutiplier)

    # decide batch number and batch size according to memory requirement
    memory_limit = gv.MEM_LIM
    batch_size = np.floor(memory_limit / (ImagePatchWidth**2*3) / 4 / 3)
    batch_num  = int(np.ceil(
        np.true_divide(testDS.xlength * testDS.ylength, batch_size)))
   
    # labeling
    _y = testDS.labels                                 # correct labels
    y  = np.zeros((testDS.num_examples,1))             # label results
    PROGRESS = progress.progress(0, batch_num)
    for j in range(batch_num):
        PROGRESS.setCurrentIteration(j+1)
        PROGRESS.setInfo(prefix_info = 'Labeling ... ', suffix_info = filename)
        PROGRESS.printProgress()
        start = testDS.index_in_epoch
        if (start + batch_size > testDS.num_examples) :
            end = testDS.num_examples
        else:
            end = start + batch_size
        batch_images, _ = testDS.next_batch(end-start)        
        y[int(start):int(end)] = classifier.predict(
            batch_images, batch_size = 256)
   
    # benchmark
    correctNumber =  np.array([
        np.sum(y == _y),                              # all correct number 
        np.sum(np.all([y == _y, _y == 0], axis = 0)), # correct negtive
        np.sum(np.all([y == _y, _y >  0], axis = 0)), # correct positive
        np.sum(np.absolute(np.subtract(y, _y)))])     # distance   
    totalInstanceNumber = _y.size                     # number of instances
    
    # save image
    image_data = np.reshape(y, (testDS.ylength, testDS.xlength))
    if(gv.dp_image_save and save_image):       
        img_save = image_data - np.amin(image_data)
        img_save = img_save / np.amax(img_save)
        io.imsave(gv.__DIR__ + gv.dp__image_dir + filename, img_save)
  
    # show image
    if(plot_show == 1):
        fig, ax = plt.subplots(1,2)
        ax[0].set_title('Original Image')
        img = io.imread(gv.__DIR__ + gv.__TrainImageDir__ + filename)
        ax[0].imshow(img)
        ax[1].set_title('Labeling Result')
        ax[1].imshow(image_data)
        plt.show()
    
    return [np.sum(y), correctNumber, totalInstanceNumber]

def testall(classifier,
            ImagePatchWidth = 20,
            ImagePatchStep = 4,
            label_mutiplier = 1.0,
            labelMode = 'PRO',
            image_show = 0,         # show labeling result of each image    
            save_image = True,      # save each labeled image
            save_result = True):    # save labeling result and accuracy 
    """ labeling all avilable images
    """
    # get all image information
    image_files, bubble_num, bubble_regions = getinfo()
    
    # files to be saved
    result_filename   = gv.dp__result_filename
    accuracy_filename = gv.dp__accuracy_filename
    
    # initialize benchmark variables
    labeling_result = np.zeros((len(image_files),1))
    correct = np.zeros(4)
    accuracy = np.zeros(4)
    totalnum_instances = 0

    for i, image_file in enumerate(image_files):
        # labeling each image
        bubbleNum, correctNumber, totalInstanceNumber = test(
            classifier,
            image_file,
            ImagePatchWidth = ImagePatchWidth,
            ImagePatchStep = ImagePatchStep,
            labelMode = labelMode,
            label_mutiplier = label_mutiplier,
            plot_show = image_show,
            save_image = save_image)
        
        labeling_result[i] = bubbleNum
        correct = np.add(correct, correctNumber)
        totalnum_instances +=  totalInstanceNumber

    # calculate total labeling accuracy            
    accuracy = np.true_divide(correct, totalnum_instances)
    
    # save data
    if(gv.dp_test_save_data and save_result):
        directory = gv.__dir__ + gv.dp__tuningPar_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        accuracy.tofile(directory + accuracy_filename, sep = " ")
        labeling_result.tofile(result_filename, sep = " ")
    return [labeling_result, accuracy]

def performance(ImagePatchWidth = 100,
                ImagePatchStep = 10,
                trainBatchNum = 10000,
                trainBatchSize = 2000,
                label_mutiplier = 1.0,
                hidden_units = [200, 400, 200],
                trainSteps = 200,
                optimizer = 'Adagrad', #"SGD", "Adam", "Adagrad"
                learning_rate = 0.01,
                label_mode = 'PRO',
                clip_gradients = 5.0,
                config = None,
                verbose = 1,
                dropout = None,
                plot_show = 1,
                save_result = True):

    """ performance benchmark through train nand labeling
    """
    # information of all aviliable images
    image_files, bubble_num, bubble_regions = getinfo()

    # train DNN
    classifier = train(
        batchNum        = trainBatchNum,
        batchSize       = trainBatchSize,
        ImagePatchWidth = ImagePatchWidth,
        ImagePatchStep  = ImagePatchStep,
        label_mutiplier = label_mutiplier,
        labelMode       = label_mode,
        hidden_units    = hidden_units,
        steps           = trainSteps,
        optimizer       = optimizer,
        learning_rate   = learning_rate,
        clip_gradients  = clip_gradients,
        config          = config,
        verbose         = verbose,
        dropout         = dropout)

    # labeling all images    
    result, accuracy = testall(
        classifier,
        ImagePatchWidth = ImagePatchWidth,
        ImagePatchStep = ImagePatchStep,
        labelMode = label_mode, 
        label_mutiplier = label_mutiplier,
        image_show = 0,
        save_image = False,
        save_result = False) # Save in this function, no need to save twice
    
    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        bubble_num, result.T)
    status  = np.append(
        np.array([slope, intercept, r_value, p_value, std_err]), accuracy)
    
    # save data
    if(save_result):
        directory = gv.__DIR__ + gv.dp__tuningPar_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = (
            str(ImagePatchWidth)  + '_' +
            str(ImagePatchStep)   + '_' +
            str(trainBatchNum)    + '_' +
            str(trainBatchSize)   + '_' +
            str(label_mutiplier)  + '_' +
            str(hidden_units)     + '_' +
            str(trainSteps)       + '_' +
            optimizer             + '_' +
            str(learning_rate)    + '_' +
            label_mode            + '_' +
            str(clip_gradients)   + '_' +
            str(config)           + '_' +
            str(verbose)          + '_' +
            str(dropout)          + '.dat')
        result.tofile(directory + filename)
    
    # show linearity
    if(plot_show == 1):
        print accuracy
        yfit = slope * bubble_num + intercept    
        fig, ax = plt.subplots(1,1)
        ax.set_title('Linearity of DNN Regressor')
        ax.set_xlabel('Bubble number counted manually')
        ax.set_ylabel('Bubble number labeled by DNN')
        ax.scatter(bubble_num, result, color = 'blue', label = 'DNN result')
        ax.plot(bubble_num, yfit, color = 'red', label = 'linear fit')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left")    
        text_x = np.amax(bubble_num)*0.6
        text_y = np.amax(result)*0.1
        text   = "R_SQUARED: %.5f\nSTD_ERR: %.5f" % (r_value**2, std_err)
        ax.annotate(text, xy = (text_x, text_y), xytext = (text_x, text_y))
        plt.show()

    return [result, status]

def multi_run_wrapper(args):
    return performance(*args)

def tuningParameters( MaxProcessNum = 8,
                      trainBatchNum = [10000],
                      trainBatchSize = [2000],
                      trainSteps = [200],
                      label_mutipliers = [1.0],
                      optimizer = ['Adagrad'], #"SGD", "Adam", "Adagrad"
                      learning_rate = [0.001, 0.005, 0.01, 0.05],
                      ImagePatchWidth = [10, 20, 50, 100, 150],
                      ImagePatchStep = [5],
                      label_mode =['PRO', 'NUM'],
                      hidden_units = [[200, 400, 200]],
                      clip_gradients = [5.0],
                      dropout = [None],
                      verbose = 1,
                      plot_show = 1,
                      save_result = True):
    """
    Tune parameter for optimization
      MaxProcessNum:  max number of processes for paralelle. if value < 1,
        program will use all the possible resouces (cpu / gpu)
   
    """
    # close all unnecessary process
    gv.dp_test_save_data = False
    gv.dp_image_save = False
    gv.log_write = False
    gv.show_progress = False

    # get all possible parameters
    pars  = []

    for hu in hidden_units:
     for bn in trainBatchNum:
      for bs in trainBatchSize:
       for ts in trainSteps:
        for op in optimizer:
         for lm in label_mode:
          for pw in ImagePatchWidth:
           for ps in ImagePatchStep:
            for cg in clip_gradients:
             for lr in learning_rate:
              for dr in dropout:
               for lms in label_mutipliers:
                pars.append((pw, ps, bn, bs, lms, hu, ts, op, lr, lm, cg, None, 
                             verbose, dr, 0, True))

    # benchmark for all possible parameters
    if(MaxProcessNum <= 0):
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(processes = MaxProcessNum)
    evaluation = pool.map(multi_run_wrapper, pars)
    pool.close()
    pool.join()

    # save data
    if(save_result):
        FILE_HEAD = ['ImagePatchWidth',
                     'ImagePatchStep',
                     'trainBatchNum',
                     'trainBatchSize',
                     'label_mutiplier',
                     'hidden_units',
                     'trainSteps',
                     'optimizer',
                     'learning_rate',
                     'label_mode',
                     'clip_gradients',
                     'dropout',
                     'slope',
                     'intercept',
                     'r_value',
                     'p_value',
                     'std_err',
                     'accuracy_total',
                     'accuracy_neg',
                     'accuracy_pos',
                     'accuracy_distance']
        directory = gv.__DIR__ + gv.dp__tuningPar_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(directory + gv.dp__tuningPar_filename, "w")
        for item in FILE_HEAD:
            f.write(item + ' ')
        f.write('\n')
        for par, eval_res in zip(pars, evaluation):
            f.write(str(par))
            for item in eval_res[-1]:
                f.write(' %f' % item)
            f.write('\n')  
        f.close()

    #plot result
    if(plot_show == 1):
        r_squared = np.zeros(len(evaluation))
        for i, eval_res in enumerate(evaluation):
            r_squared[i] = eval_res[-1][2] 
        fig, ax = plt.subplots(1, 1)
        ax.plot(r_squared)
        ax.set_xlabel('Parameter index')
        ax.set_ylabel('R_Squared')
        plt.show()
    return [pars, evaluation]

def plotresultfromfile(ImagePatchWidth = 100,
                       ImagePatchStep = 10,
                       trainBatchNum = 10000,
                       trainBatchSize = 2000,
                       label_mutiplier = 1.0,
                       hidden_units = [200, 400, 200],
                       trainSteps = 200,
                       optimizer = 'Adagrad', #"SGD", "Adam", "Adagrad"
                       learning_rate = 0.01,
                       label_mode = 'PRO',
                       clip_gradients = 5.0,
                       config = None,
                       verbose = 1,
                       dropout = None):
    """
    load data from file and show the data
    """
    image_files, bubble_num, bubble_regions = getinfo()
    directory = gv.__DIR__ + gv.dp__tuningPar_dir
    filename = (
        str(ImagePatchWidth)  + '_' +
        str(ImagePatchStep)   + '_' +
        str(trainBatchNum)    + '_' +
        str(trainBatchSize)   + '_' +
        str(label_mutiplier)  + '_' +
        str(hidden_units)     + '_' +
        str(trainSteps)       + '_' +
        optimizer             + '_' +
        str(learning_rate)    + '_' +
        label_mode            + '_' +
        str(clip_gradients)   + '_' +
        str(config)           + '_' +
        str(verbose)          + '_' +
        str(dropout)          + '.dat')
    result = np.fromfile(directory + filename)
    slope, intercept, r_value, p_value, std_err = linregress(
        bubble_num, result.T)
    yfit = slope * bubble_num + intercept    
    fig, ax = plt.subplots(1,1)
    ax.set_title('Linearity of DNN Regressor')
    ax.set_xlabel('Bubble number counted manually')
    ax.set_ylabel('Bubble number labeled by DNN')
    ax.scatter(bubble_num, result, color = 'blue', label = 'DNN result')
    ax.plot(bubble_num, yfit, color = 'red', label = 'linear fit')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left")    
    text_x = np.amax(bubble_num)*0.6
    text_y = np.amax(result)*0.2
    text   = "R_SQUARED: %.5f\nSTD_ERR: %.5f" % (r_value**2, std_err)
    ax.annotate(text, xy = (text_x, text_y), xytext = (text_x, text_y))
    plt.show()

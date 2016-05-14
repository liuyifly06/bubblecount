import sys, traceback
import numpy as np
import bubblecount.globalvar as gv
from bubblecount.neuralnetwork import deeplearning as dl
from bubblecount.preprocess.readinfo import getinfo
from bubblecount.neuralnetwork import dataset as ds
from bubblecount.progressbar import progress
from scipy.stats import linregress
from matplotlib import pyplot as plt

def main():
    try:
        echos_tune = 10
        batchNum = 2000
        batchSize = 2000
        ImagePatchWidth = 40
        ImagePatchStep = 1
        labelMode = 'NUM'
        label_mutiplier = 1.0
        hidden_units = [200, 400, 200]
        steps = 200
        optimizer = 'Adagrad' #"SGD", "Adam", "Adagrad"
        learning_rate = 0.1
        clip_gradients = 5.0
        config = None
        verbose = 1
        dropout = None
        filename = 'detector_3_no_5_angle_2.jpg'
       
        image_files, bubble_num, bubble_regions = getinfo()        
        benchmark_tune = np.zeros((echos_tune,2))

        # generate data:        
        #ds.gaussianDatatoFile(ImagePatchWidth, ImagePatchStep, labelMode)
    
        classifier, trainDataset = dl.train(
            batchNum = 0,
            batchSize = 1,
            ImagePatchWidth = ImagePatchWidth,
            ImagePatchStep = ImagePatchStep,
            labelMode = labelMode,
            hidden_units = hidden_units,
            steps = steps,
            optimizer = optimizer,
            learning_rate = learning_rate,
            clip_gradients = clip_gradients,
            config = config,
            verbose = verbose,
            dropout = dropout)
        PROGRESS = progress.progress(0, echos_tune)
        

        for i in range(echos_tune):
            PROGRESS.setCurrentIteration(i+1)
            PROGRESS.setInfo(prefix_info = 'Batch Number Tune ...',
                             suffix_info = str(i+1) + '/' +str(echos_tune))
            classifier = dl.continuetrain(
                classifier = classifier,
                trainDataset = trainDataset,
                batchNum = batchNum,
                batchSize = batchSize)
            result, accuracy = dl.testall(classifier,
                ImagePatchWidth = ImagePatchWidth,
                ImagePatchStep = ImagePatchStep,
                label_mutiplier = label_mutiplier,
                labelMode = labelMode,
                image_show = 0,     # show labeling result of each image    
                save_image = False,      # save each labeled image
                save_result = False)
            slope, intercept, r_value, p_value, std_err = linregress(
                bubble_num, result.T)
            benchmark_tune[i,0] = r_value
            benchmark_tune[i,1] = std_err
            PROGRESS.printProgress()
            print '\n'
        filename = gv.__DIR__ + gv.dp__tuningPar_dir + 'batchnumberresult.npy'
        np.save(filename, np.array(benchmark_tune))
        fig, ax = plt.subplots(1)
        ax.plot(np.arange(echos_tune), benchmark_tune[:,0], color = 'r', label = 'r_value')
        ax.plot(np.arange(echos_tune), benchmark_tune[:,1], color = 'b', label = 'std_err')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left")  
        plt.show()
        
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

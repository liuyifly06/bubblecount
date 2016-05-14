import sys, traceback
from bubblecount.neuralnetwork import deeplearning as dl
def main():
    try:
        batchNum = 2000
        batchSize = 20000
        ImagePatchWidth = 40
        ImagePatchStep = 1
        labelMode = 'PRO'
        hidden_units = [200, 400, 200]
        steps = 1
        optimizer = 'Adagrad' #"SGD", "Adam", "Adagrad"
        learning_rate = 0.14
        clip_gradients = 5.0
        config = None
        verbose = 1
        dropout = None
        filename = 'detector_3_no_5_angle_2.jpg'

        classifier =dl.train(
            batchNum = batchNum,
            batchSize = batchSize,
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
        
        dl.test(
            classifier,
            filename,
            ImagePatchWidth = ImagePatchWidth,
            ImagePatchStep = ImagePatchStep,
            labelMode = labelMode,
            plot_show = 1,
            save_image = True)
        #"""
        """
        dl.performance(
            ImagePatchWidth = ImagePatchWidth,
            ImagePatchStep = ImagePatchStep,
            trainBatchNum = batchNum,
            trainBatchSize = batchSize,
            hidden_units = hidden_units,
            trainSteps = steps,
            optimizer = optimizer,
            learning_rate = learning_rate,
            label_mode = labelMode,
            clip_gradients = clip_gradients,
            config = config,
            verbose = verbose,
            dropout = dropout,
            plot_show = 1,
            save_result = True)
        #"""
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

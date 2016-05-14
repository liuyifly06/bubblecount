import sys, traceback
from bubblecount.neuralnetwork import deeplearning as dl
def main():
    try:
        label_mutiplier = 1.0,
        batchNum = 1
        batchSize = 2
        ImagePatchWidth = 10
        ImagePatchStep = 1000
        labelMode = 'PRO'
        hidden_units = [200, 400, 200]
        steps = 1
        optimizer = 'Adagrad' #"SGD", "Adam", "Adagrad"
        learning_rate = 0.1
        clip_gradients = 5.0
        config = None
        verbose = 1
        dropout = None
        filename = 'detector_3_no_5_angle_2.jpg'
        """
        classifier =dl.train(
            batchNum = batchNum,
            batchSize = batchSize,
            ImagePatchWidth = ImagePatchWidth,
            ImagePatchStep = ImagePatchStep,
            label_mutiplier = label_mutiplier,
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
            label_mutiplier = label_mutiplier,
            ImagePatchWidth = ImagePatchWidth,
            ImagePatchStep = ImagePatchStep,
            labelMode = labelMode,
            plot_show = 1,
            save_image = True)
       
        dl.testall(
            classifier,
            ImagePatchWidth = ImagePatchWidth,
            ImagePatchStep = ImagePatchStep,
            label_mutiplier = label_mutiplier,
            labelMode = labelMode) 
        
        dl.performance(
            ImagePatchWidth = ImagePatchWidth,
            ImagePatchStep = ImagePatchStep,
            trainBatchNum = batchNum,
            trainBatchSize = batchSize,
            label_mutiplier = label_mutiplier,
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
        """
        """
        dl.tuningParameters(
            MaxProcessNum = 8,
            trainBatchNum = [batchNum],
            trainBatchSize = [batchSize],
            trainSteps = [steps],
            label_mutipliers = [label_mutiplier],
            optimizer = [optimizer], #"SGD", "Adam", "Adagrad"
            learning_rate = [learning_rate],
            ImagePatchWidth = [ImagePatchWidth],
            ImagePatchStep = [ImagePatchStep],
            label_mode =['PRO', 'NUM'],
            hidden_units = [hidden_units],
            clip_gradients = [clip_gradients],
            dropout = [dropout],
            verbose = 1,
            plot_show = 1,
            save_result = True)
        #"""
        dl.plotresultfromfile(
            ImagePatchWidth = ImagePatchWidth,
            ImagePatchStep = ImagePatchStep,
            trainBatchNum = batchNum,
            trainBatchSize = batchSize,
            label_mutiplier = label_mutiplier,
            hidden_units = hidden_units,
            trainSteps = steps,
            optimizer = optimizer,
            learning_rate = learning_rate,
            label_mode = labelMode,
            clip_gradients = clip_gradients,
            config = config,
            verbose = verbose,
            dropout = dropout)
        #"""
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

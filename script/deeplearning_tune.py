import sys, traceback
from bubblecount.neuralnetwork import deeplearning as dl
def main():
    try:
        dl.tuningParameters(
            MaxProcessNum = 24,
            trainBatchNum = [20000],
            trainBatchSize = [20000],
            trainSteps = [200],
            label_mutipliers = [1.0, 100.0],
            optimizer = ["Adagrad"], #"SGD", "Adam", "Adagrad"
            learning_rate = [0.1],
            ImagePatchWidth = [40],
            ImagePatchStep = [1],
            label_mode =['PRO', 'NUM'],
            hidden_units = [[50,100,50],[200,400,200]],
            clip_gradients = [clip_gradients],
            dropout = [None, 0.2, 0.5],
            verbose = 1,
            plot_show = 1,
            save_result = True)
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

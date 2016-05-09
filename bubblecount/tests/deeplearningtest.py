import sys, traceback
from bubblecount.neuralnetwork import deeplearning as dl
def main():
    try:      
        if len(sys.argv) > 1:
            print('Runing ')
            dl.test_run(ins_size = 40,
                stride = 1,
                label_option = 0,
                batch_num = 20000,
                batch_size = 20000,
                learning_rate = 0.14,
                label_mode = 'PRO',
                run_mode = sys.argv[1])
        else:
            print ('Parameter Tuning')
            dl.tuningParameters( MaxProcessNum = 2,
                                 batch_num = [1],
                                 batch_size = [200],
                                 learning_rate = [0.5],
                                 ins_size = [20, 20],
                                 stride = [40],
                                 label_option = [100],
                                 label_mode =['PRO'] )
            """
            dl.tuningParameters(batch_num = 10000,
                      batch_size = 2000,
                      learning_rate = [0.001, 0.01, 0.05],
                      ins_size = [20, 50, 100, 200],
                      stride = [10, 20],
                      label_option = [100, 1000],
                      label_mode =['PRO', 'NUM'] )
            """
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

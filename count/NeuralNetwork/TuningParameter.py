import sys, traceback, multiprocessing
import numpy as np
from scipy.stats import linregress
from .. import GlobalVariables as gv
from ..PreProcess.readinfo import getInfo
from ..ProgressBar import progress
import deepLearning as dl

def multi_run_wrapper(args):
    return dl.performance(*args)

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
                  info.append('batch:' + str(bn) + '_' + str(bs) +
                          ' patch:' + str(ins)+ '_' + str(s)  +
                          ' train:' + str(lo) + '_' + str(lr) + '_' + lm)
                  pars.append((ins, s, lo, bn, bs, lr, lm))

    # calculating with all the parameters
    if(MaxProcessNum <= 0):
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(processes = MaxProcessNum)
    evaluation = pool.map(multi_run_wrapper, pars)
    print [len(info), len(evaluation)]
    f = open(gv.__DIR__ + gv.dp__tuningPar_dir 
             + gv.dp__tuningPar_filename, "w")
    for info_line, eval_res in zip(info, evaluation):
        f.write(info_line + ' ' + np.array_str(eval_res) + '\n')  
    f.close()
    return [info, evaluation]

                        
def main():
    try:      
        tuningParameters( MaxProcessNum = 20,
                          batch_num = [10000],
                          batch_size = [2000],
                          learning_rate = [0.001, 0.005, 0.01, 0.05],
                          ins_size = [150, 100, 50, 20],
                          stride = [10, 20],
                          label_option = [100],
                          label_mode =['PRO', 'NUM'] )
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

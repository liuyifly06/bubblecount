import sys, traceback, multiprocessing, os
import numpy as np
from scipy.stats import linregress
from bubblecount import globalvar as gv
from bubblecount.preprocess.readinfo import getinfo
from bubblecount.progressbar import progress
import deeplearning as dl

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

                        
def main():
    try:      
        tuningParameters( MaxProcessNum = 2,
                          batch_num = [1],
                          batch_size = [200],
                          learning_rate = [0.5],
                          ins_size = [20, 20],
                          stride = [40],
                          label_option = [100],
                          label_mode =['PRO'] )
        """
        tuningParameters( MaxProcessNum = 18,
                          batch_num = [10000],
                          batch_size = [2000],
                          learning_rate = [0.11, 0.12, 0.13, 0.14],
                          ins_size = [40, 50, 60],
                          stride = [7, 10, 15],
                          label_option = [100],
                          label_mode = ['PRO'] )
        """
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

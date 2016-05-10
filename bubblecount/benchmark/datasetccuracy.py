# accuracy measurement method for generating instances 
import multiprocessing, os
import numpy as np
from bubblecount.neuralnetwork import dataset as ds
from bubblecount.preprocess.readinfo import getinfo
from bubblecount.progressbar import progress
from bubblecount import globalvar as gv
from matplotlib import pyplot as plt
from scipy.stats import linregress

def labellinearity(patch_size, stride, numOfClasses, labelMode,
                   progress_show = 1, plot_show = 1):
    gv.ds_show_filename = False 
    image_files, bubble_num, bubble_regions = getinfo()
    bubble_num_afterlabel  =  np.zeros(len(image_files))
    probar = progress.progress(0, len(image_files))

    for i, image in enumerate(image_files):
        probar.setCurrentIteration(i+1)
        probar.setInfo(prefix_info = 'dataset linearity ...',
                       suffix_info = image)
        probar.printProgress()  
        image_ds = ds.read_data_sets(patch_size, stride, numOfClasses, 'test', 
                   labelMode, imageName = image)
        labels = image_ds.labels
        bubble_num_afterlabel[i] = np.sum(labels)
 

    slope, intercept, r_value, p_value, std_err = (linregress(bubble_num,
                                                      bubble_num_afterlabel))

    if(plot_show == 1):    
        yfit = slope * bubble_num + intercept    
        fig, ax = plt.subplots(1,1)
        ax.set_title('Linearity of Labeling Methods')
        ax.set_xlabel('Bubble number counted manually')
        ax.set_ylabel('Bubble number after labeling')
        ax.scatter(bubble_num, bubble_num_afterlabel,
                   color = 'blue', label = 'bubble number')
        ax.plot(bubble_num, yfit, color = 'red', label = 'linear regression')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left")    
        text_x = np.amax(bubble_num)*0.6
        text_y = np.amax(bubble_num_afterlabel)*0.1
        text   = "r_squared: %.5f\nstd_err: %.5f" % (r_value**2, std_err)
        ax.annotate(text, xy = (text_x, text_y), xytext = (text_x, text_y))
        plt.show()
    
    return ([[slope, intercept, r_value, p_value, std_err],
             bubble_num, bubble_num_afterlabel])

def multi_run_wrapper(args):
    return labellinearity(*args)

def labellinearity_stride(patch_sizes, strides, numOfClasses, labelModes,
                          MaxProcessNum = -1):

    # stop showing process bar for mutilprocessing
    gv.show_progress = False  

    pars  = []
    for patch_size in patch_sizes:
      for stride in strides:
        for numclass in numOfClasses:
          for labelMode in labelModes:
            pars.append((patch_size, stride, numclass, labelMode, 0, 0))

    if(MaxProcessNum <= 0):
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(processes = MaxProcessNum)

    result_linearity = pool.map(multi_run_wrapper, pars)
    pool.close()
    pool.join()
    r_squared = [] 
    directory = gv.__DIR__ + gv.label_linearity_file_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    fid = open(directory + gv.label_linearity_file, "w")
    for par, res in zip(pars, result_linearity):      
        par = list(par)
        if par[3] == 'PRO':
            par[3] = 0
        elif par[3] == 'NUM':
            par[3] = 1
        for val in par:
            fid.write('%d ' % val)
        fid.write('%f\n' % res[0][2])
        r_squared.append(par + [res[0][2]])
    fid.close()
    #restore showing process bar status
    gv.show_progress = True
    return np.asarray(r_squared)

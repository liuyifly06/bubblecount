import sys, traceback, time
import numpy as np
import preprocessing as pre
import trainGenerateInstances as trgi
import testGenerateInstances as tegi

from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import canny

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, \
        LinearLayer, SigmoidLayer, FullConnection

from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData

def output_data():
    target = np.array([1, 27, 40, 79, 122, 160, \
                       1, 18, 28, 42, 121, 223, \
                       0, 11, 24, 46, 142, 173, \
                       3, 19, 23, 76, 191, 197, \
                       0, 15, 24, 45,  91, 152,  \
                       0, 16, 27, 34,  88,      \
                       0,  9, 12, 69, 104, 123])
    result = np.repeat(target,3)
    return result.T

def input_data(DETECTOR_NUM = 7, TRAILS = 6, ANGLE_NUM = 3):
    index = -1
    data = np.array([])
    for det in range(1,DETECTOR_NUM+1):
        if (det!=6):
            num_n = TRAILS
        else:
            num_n = TRAILS - 1
            
        for n in range(0,num_n):
            
            for angle in range(1, ANGLE_NUM+1):
                index = index + 1
                filename = 'detector_' + str(det) + '_no_' + str(n) \
                        + '_angle_' + str(angle) + '.jpg'
                
                print 'Reading data %s ... %.2f' \
                        % (filename, 100 * index / \
                        ((DETECTOR_NUM * TRAILS - 1) * ANGLE_NUM)) \
                        + "% finishied"
                image = io.imread(filename)
                image_gray = rgb2gray(image)

                input_instance = np.reshape(image_gray,(1,image_gray.shape[0]*image_gray.shape[1]))
                
                if(data.size>0):
                    data = np.append(data, input_instance, axis=0)
                else:
                    data = input_instance
                    
#    np.savetxt('data.txt', data, delimiter=',')
    return data

def input_data_fromfile(filename):
    return numpy.loadtxt(filename)
    
def GenerateDataSet(ia, ta):
    ds = SupervisedDataSet(ia.shape[1], 1)
    assert(ia.shape[0] == ta.shape[0])
    for i in range(0,ia.shape[0]):
        ds.addSample(ia[i,:], ta[i])
    #ds.setField('input',  ia)
    #ds.setField('target', ta)
    #TrainDS, TestDS = ds.splitWithProportion(0.8)
    return ds

def build_network(input_dim,layers,output_dim):
    
    n = FeedForwardNetwork()
    inLayer = LinearLayer(input_dim)
    hiddenLayer = SigmoidLayer(layers)
    outLayer = LinearLayer(output_dim)
    
    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)
    
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)
    
    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)
    n.sortModules()
    
    return n

def main():
    try:
        start_time = time.time()
        elapse_time = time.time() - start_time
        print 'Start reading images to generate data ...' \
                +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                + ' has past. '

        #Read Data
        input_ds = input_data(2, 6, 3)
        output_ds = output_data()
        output_ds = output_ds[0:36]
        dataset = GenerateDataSet(input_ds, output_ds)
        TrainDS, TestDS = dataset.splitWithProportion(0.8)
        
        elapse_time = (time.time()-start_time)
        print 'Start training neutral neutwrok ...' \
                +  time.strftime(" %H:%M:%S", time.gmtime(elapse_time)) \
                + ' has past. '
        
        #Build Netwrok
        LAYERS_NUM = 5
        net = build_network(dataset['input'].shape[1],LAYERS_NUM,1)
        # net = buildNetwork(m*n, LAYERS_NUM, 1)
        
        #Training
        trainer = BackpropTrainer(net, dataset=TrainDS)
        trainer.trainUntilConvergence()
        
        #Test
        for inp, targ in TrainDS:
            print [net.activate(inp), targ]
        
        print '_________________________________'
        for inp, targ in TestDS:
            print [net.activate(inp), targ]

        trainer.testOnClassData(TestDS)
        
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

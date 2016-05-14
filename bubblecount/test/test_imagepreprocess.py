import sys, traceback, time
from bubblecount.preprocess import preprocessing as pp
def main():
    try:
        # Constants
        Window_Size = 5
        # Default Parameters        
        if len(sys.argv) >= 2:
            print '\n Test Mode \n'
            pp.test(sys.argv[1], sys.argv[2])
            """
            test(bubble_filename, background_filename, Window_Size = 5, plot_image = 1,
                 pars = [1, 600, 50, 0, 0.5, 0.5]):
            """   
        else:
            print 'Batch Processing ...'
            pp.batch_process(pars = [0, 600, 50, 20, 0.5, 0.5], label = False)
            """
            batch_process(pars = [0, 630, 50, 40, 0.5, 0.5])
            """
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

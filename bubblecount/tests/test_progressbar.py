import sys, traceback, time
from bubblecount.progressbar import progress
def main():
    try:
        n = 100
        test = progress.progress(0,n)        
        for i in range(n):
            test.setCurrentIteration(i+1)
            test.setInfo(prefix_info = 'Progress Bar Example ... ',
                         suffix_info = 'Iteration '+str(i)+'/'+str(n))
            test.printProgress()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

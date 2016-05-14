import sys, traceback
import bubblecount.neuralnetwork.dataset as ds
def main():
    try:
        ds.gaussianData2File();
    except KeyboardInterrupt:
        print "Shutdown requested... exiting"
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == '__main__':
    main()

from keras import backend
from utility.enums import Processor

def Configure(args):    
    import keras as keras
    print("keras version: " + keras.__version__)

    backendname = backend.backend()
    if backendname == 'cntk':
        __configure_CNTK(args)
    elif backendname == 'tensorflow':
        __configure_tensorflow(args)
    elif backendname == 'theano':
        __configure_Theano(args)
    else:
        raise ValueError("unable to detect backend for configuration")

def Clean_Session():
    backendname = backend.backend()
    if backendname == 'cntk':
        __clean_session_cntk()
    elif backendname == 'tensorflow':
        __clean_session_tensorflow()
    elif backendname == 'theano':
        __clean_session_theano()
    else:
        raise ValueError("unable to detect backend for cleanup")
    # cleanup
    from keras import backend as K
    K.clear_session()

def __configure_Theano(args):
    print("theano configured")

def __configure_CNTK(args):
    print("cntk configured")

def __configure_tensorflow(args):
    import tensorflow as tf
    print("tensorflow version: " + tf.__version__)
    tf.logging.set_verbosity(tf.logging.INFO)
    Clean_Session()

    if(args['processor'] != Processor.GPU):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    if(args['processor'] == Processor.TPU):
        # setup tpu worker for google colab (colab.research.google.com)
        tpu_addr = os.environ.get('COLAB_TPU_ADDR')
        if(tpu_addr is not None):
            args['TPU_WORKER'] = 'grpc://' + tpu_addr
            print("tpu worker: {}".format(args['TPU_WORKER']))
        else:
            args['processor'] = Processor.CPU
            print("no tpu worker found, falling back to cpu")
    elif(args['processor'] == Processor.GPU):
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras
        print("multi process gpu usage: enabled")
    print("tensorflow configured")

def __clean_session_theano():
    print("theano session cleaned")

def __clean_session_cntk():
    print("cntk session cleaned")

def __clean_session_tensorflow():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    print("tensorflow session cleaned")

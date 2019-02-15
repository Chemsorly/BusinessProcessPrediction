from utility.enums import Processor, RnnType

def CreateModel(args):
    keras_impl = __getKerasImplementation(args['processor'])
    if(args['processor'] == Processor.TPU):
        print("imported tensorflow.keras API for model creation")
    else:
        print("imported keras API for model creation")

    if args['rnntype'] == RnnType.LSTM :
        if args['processor'] == Processor.GPU and args['cudnn'] == True:
            print("creating stateless cudnn lstm model")
            return __createCUDNN_LSTM_Stateless(keras_impl,args) 
        else:
            print("creating stateless lstm model")
            return __createLSTM_Stateless(keras_impl,args)  
    elif args['rnntype'] == RnnType.GRU:
        if args['processor'] == Processor.GPU and args['cudnn'] == True:
            print("creating stateless cudnn gru model")
            return __createCUDNN_GRU_Stateless(keras_impl,args) 
        else:
            print("creating stateless gru model")
            return __createGRU_Stateless(keras_impl,args)   
    elif args['rnntype'] == RnnType.RNN:
        print("creating stateless rnn model")
        return __createRNN_Stateless(keras_impl,args) 
    else:
        raise ValueError("unkown model type")

def CreateCallbacks(args):
    callbacks = []
    keras_impl = __getKerasImplementation(args['processor'])

    callbacks.append(keras_impl.callbacks.EarlyStopping(monitor='val_loss', patience=args['patience_earlystopping']))
    if(args['save_model'] == True):
        callbacks.append(keras_impl.callbacks.ModelCheckpoint('{}-model.h5'.format(args['running']), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto'))
    if(args['processor'] != Processor.TPU):
        callbacks.append(keras_impl.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=args['patience_reducelr'], verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))
    callbacks.append(keras_impl.callbacks.CSVLogger('{}-epochlogs.epochlog'.format(args['running'])))
    if args['tensorboard'] == True:
        callsbacks.append(tensorboard_cb = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True))
    return callbacks

def CreateOptimizer(keras_impl, args):
    if args['processor'] == Processor.TPU:
        import tensorflow as tf 
        return tf.contrib.opt.NadamOptimizer(learning_rate=args['learningrate'], beta1=0.9, beta2=0.999, epsilon=1e-08)
    else:
        return keras_impl.optimizers.Nadam(lr=args['learningrate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=args['gradientclipvalue'])   

def __getKerasImplementation(processor):
    if(processor == Processor.TPU):
        import tensorflow
        from tensorflow.python import keras as keras_impl
    else:
        import keras as keras_impl
    return keras_impl

#lstm
def __createCUDNN_LSTM_Stateless(keras_impl,args):
    model = keras_impl.models.Sequential()
    if args['layers'] == 1:
        if args['bidirectional'] == True:
            model.add(keras_impl.layers.Bidirectional(keras_impl.layers.CuDNNLSTM(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform'),input_shape=(args['maxlen'],args['num_features']))) 
        else:
            model.add(keras_impl.layers.CuDNNLSTM(args['neurons'],input_shape=(args['maxlen'],args['num_features']),return_sequences=False, kernel_initializer='glorot_uniform'))        
    if args['layers'] > 1:
        if args['bidirectional'] == True:            
            model.add(keras_impl.layers.Bidirectional(keras_impl.layers.CuDNNLSTM(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform'),input_shape=(args['maxlen'],args['num_features'])))
        else:
            model.add(keras_impl.layers.CuDNNLSTM(args['neurons'],input_shape=(args['maxlen'],args['num_features']),return_sequences=True, kernel_initializer='glorot_uniform'))
        for i in range(args['layers'] - 1):
            model.add(keras_impl.layers.BatchNormalization())
            if i == args['layers'] - 2:
                if args['bidirectional'] == True:
                    model.add(keras_impl.layers.Bidirectional(keras_impl.layers.CuDNNLSTM(args['neurons'], return_sequences=False, kernel_initializer='glorot_uniform'))) 
                else:
                    model.add(keras_impl.layers.CuDNNLSTM(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform'))                  
            else:
                if args['bidirectional'] == True: 
                    model.add(keras_impl.layers.Bidirectional(keras_impl.layers.CuDNNLSTM(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform')))
                else:
                    model.add(keras_impl.layers.CuDNNLSTM(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform'))
    model.add(keras_impl.layers.BatchNormalization())
    model.add(keras_impl.layers.Dense(1, kernel_initializer='glorot_uniform', name='time_output'))

    opt = CreateOptimizer(keras_impl, args)
    model.compile(loss={'time_output':'mae'}, optimizer=opt)
    return model

def __createLSTM_Stateless(keras_impl,args):
    model = keras_impl.models.Sequential()
    if args['layers'] == 1:
        if args['bidirectional'] == True:
            model.add(keras_impl.layers.Bidirectional(keras_impl.layers.LSTM(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout']),input_shape=(args['maxlen'],args['num_features'])))
        else:
            model.add(keras_impl.layers.LSTM(args['neurons'],input_shape=(args['maxlen'],args['num_features']),return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout']))
    if args['layers'] > 1:
        if args['bidirectional'] == True:
            model.add(keras_impl.layers.Bidirectional(keras_impl.layers.LSTM(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout']),input_shape=(args['maxlen'],args['num_features'])))
        else:
            model.add(keras_impl.layers.LSTM(args['neurons'],input_shape=(args['maxlen'],args['num_features']),return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout']))
        for i in range(args['layers'] - 1):
            model.add(keras_impl.layers.BatchNormalization())
            if i == args['layers'] - 2:
                if args['bidirectional'] == True: 
                    model.add(keras_impl.layers.Bidirectional(keras_impl.layers.LSTM(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout'])))
                else:
                    model.add(keras_impl.layers.LSTM(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout']))                
            else:
                if args['bidirectional'] == True: 
                    model.add(keras_impl.layers.Bidirectional(keras_impl.layers.LSTM(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout'])))
                else:
                    model.add(keras_impl.layers.LSTM(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout']))
    model.add(keras_impl.layers.BatchNormalization())
    model.add(keras_impl.layers.Dense(1, kernel_initializer='glorot_uniform', name='time_output'))

    opt = CreateOptimizer(keras_impl, args)   
    model.compile(loss={'time_output':'mae'}, optimizer=opt)
    return model

#gru
def __createCUDNN_GRU_Stateless(keras_impl,args):
    model = keras_impl.models.Sequential()
    if args['layers'] == 1:
        if args['bidirectional'] == True:            
            model.add(keras_impl.layers.Bidirectional(keras_impl.layers.CuDNNGRU(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform'),input_shape=(args['maxlen'],args['num_features'])))
        else:
             model.add(keras_impl.layers.CuDNNGRU(args['neurons'],input_shape=(args['maxlen'],args['num_features']),return_sequences=False, kernel_initializer='glorot_uniform'))       
    if args['layers'] > 1:
        if args['bidirectional'] == True:            
            model.add(keras_impl.layers.Bidirectional(keras_impl.layers.CuDNNGRU(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform'),input_shape=(args['maxlen'],args['num_features'])))
        else:
            model.add(keras_impl.layers.CuDNNGRU(args['neurons'],input_shape=(args['maxlen'],args['num_features']),return_sequences=True, kernel_initializer='glorot_uniform'))
        for i in range(args['layers'] - 1):
            model.add(keras_impl.layers.BatchNormalization())
            if i == args['layers'] - 2:
                if args['bidirectional'] == True:            
                    model.add(keras_impl.layers.Bidirectional(keras_impl.layers.CuDNNGRU(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform')))
                else:
                    model.add(keras_impl.layers.CuDNNGRU(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform'))                  
            else:
                if args['bidirectional'] == True: 
                    model.add(keras_impl.layers.Bidirectional(keras_impl.layers.CuDNNGRU(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform')))
                else:
                    model.add(keras_impl.layers.CuDNNGRU(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform'))
    model.add(keras_impl.layers.BatchNormalization())
    model.add(keras_impl.layers.Dense(1, kernel_initializer='glorot_uniform', name='time_output'))

    opt = CreateOptimizer(keras_impl, args)
    model.compile(loss={'time_output':'mae'}, optimizer=opt)
    return model

def __createGRU_Stateless(keras_impl,args):
    model = keras_impl.models.Sequential()
    if args['layers'] == 1:
        if args['bidirectional'] == True:
            model.add(keras_impl.layers.Bidirectional(keras_impl.layers.GRU(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout']),input_shape=(args['maxlen'],args['num_features'])))
        else:
            model.add(keras_impl.layers.GRU(args['neurons'],input_shape=(args['maxlen'],args['num_features']),return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout']))        
    if args['layers'] > 1:
        if args['bidirectional'] == True:
            model.add(keras_impl.layers.Bidirectional(keras_impl.layers.GRU(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout']),input_shape=(args['maxlen'],args['num_features'])))
        else:
            model.add(keras_impl.layers.GRU(args['neurons'],input_shape=(args['maxlen'],args['num_features']),return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout']))
        for i in range(args['layers'] - 1):
            model.add(keras_impl.layers.BatchNormalization())
            if i == args['layers'] - 2:
                if args['bidirectional'] == True:
                    model.add(keras_impl.layers.Bidirectional(keras_impl.layers.GRU(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout'])))
                else:
                    model.add(keras_impl.layers.GRU(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout']))                
            else:
                if args['bidirectional'] == True: 
                    model.add(keras_impl.layers.Bidirectional(keras_impl.layers.GRU(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout'])))
                else:
                    model.add(keras_impl.layers.GRU(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout']))
    model.add(keras_impl.layers.BatchNormalization())
    model.add(keras_impl.layers.Dense(1, kernel_initializer='glorot_uniform', name='time_output'))

    opt = CreateOptimizer(keras_impl, args)
    model.compile(loss={'time_output':'mae'}, optimizer=opt)
    return model

#rnn (vanilla)
def __createRNN_Stateless(keras_impl,args):
    model = keras_impl.models.Sequential()
    if args['layers'] == 1:
        if args['bidirectional'] == True:
            model.add(keras_impl.layers.Bidirectional(keras_impl.layers.SimpleRNN(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout']),input_shape=(args['maxlen'],args['num_features'])))
        else:
            model.add(keras_impl.layers.SimpleRNN(args['neurons'],input_shape=(args['maxlen'],args['num_features']),return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout']))        
    if args['layers'] > 1:
        if args['bidirectional'] == True:
            model.add(keras_impl.layers.Bidirectional(keras_impl.layers.SimpleRNN(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout']),input_shape=(args['maxlen'],args['num_features'])))
        else:
            model.add(keras_impl.layers.SimpleRNN(args['neurons'],input_shape=(args['maxlen'],args['num_features']),return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout']))
        for i in range(args['layers'] - 1):
            model.add(keras_impl.layers.BatchNormalization())
            if i == args['layers'] - 2:
                if args['bidirectional'] == True:
                    model.add(keras_impl.layers.Bidirectional(keras_impl.layers.SimpleRNN(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout'])))
                else:
                    model.add(keras_impl.layers.SimpleRNN(args['neurons'],return_sequences=False, kernel_initializer='glorot_uniform', dropout=args['dropout']))                
            else:
                if args['bidirectional'] == True: 
                    model.add(keras_impl.layers.Bidirectional(keras_impl.layers.SimpleRNN(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout'])))
                else:
                    model.add(keras_impl.layers.SimpleRNN(args['neurons'],return_sequences=True, kernel_initializer='glorot_uniform', dropout=args['dropout']))
    model.add(keras_impl.layers.BatchNormalization())
    model.add(keras_impl.layers.Dense(1, kernel_initializer='glorot_uniform', name='time_output'))

    opt = CreateOptimizer(keras_impl, args)
    model.compile(loss={'time_output':'mae'}, optimizer=opt)
    return model
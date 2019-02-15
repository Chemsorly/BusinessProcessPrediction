from __future__ import print_function, division
from collections import Counter
import unicodecsv
import numpy as np
import random
import sys
import os
from os.path import basename
import copy
import time
import shutil
from datetime import datetime
from math import log

# custom imports
import utility.dataoperations as dataoperations
import utility.models as models
import utility.regularization as regularization
import utility.preprocessing as preprocessing
import utility.configuration as configuration
import utility.generator as generator
import utility.exceptions as exceptions
from utility.enums import Processor, DataGenerationPattern

def Train_And_Evaluate(**kwargs):
    try:
        args = preprocessing.Parse_Args(**kwargs)

        # get dataset specific parameters
        datadefinition = args['datadefinition']
        args['rowstructure'] = datadefinition.GetRowstructure()
        if args['eventlog'] == "": # get dataset defined in rowstructure if not explicitly supplied
            args['eventlog'] = datadefinition.GetDataset()

        #setup (e.g. backend specifics)
        configuration.Configure(args)
        
        args = __Preprocessing(args)
        model = __Train_Model(args)
        __Evaluate_Model(args, model)
    except Exception as ex:
        # catch all, throw one exception for parsing
        raise exceptions.ConductorError(ex)

def __ReadData(indata, args):
    #read from csv file
    data = dataoperations.ReadInData(args[indata],args['rowstructure'])

    # remove sequences longer than maxseqlength
    data = dataoperations.TruncateSequences(data,args['max_sequencelength'])
    return data

def __Preprocessing(args): 
    # clean datadefinion
    dataoperations.VerifyDatadefinition(args['rowstructure'])

    # read from csv file
    data = __ReadData("eventlog",args)

    # offset data (if < 0) and create divisors
    args['offsets'] = dataoperations.CreateOffsets(data)
    args['divisors'] = dataoperations.CreateDivisors(data)

    # append EOL characters
    data = dataoperations.AppendEOL(data)
    args['maxlen'] = max(map(lambda x: len(x),data[0])) - 1 # minus eol
    print('maxlen {} '.format(args['maxlen']))

    # extract corpus
    args['indices'] = dataoperations.CreateDictionaries(data,args['rowstructure'])

    # calc feature lengths
    dataoperations.CalculateFeatures(args)

    # generate folds
    intermediate_fold_data = []
    args['traindata'] = []
    args['validationdata'] = []
    args['testdata'] = []
    # first round of folds
    split_index = int(len(data[0]) - (len(data[0]) * args['testdata_split']))
    for i in range(len(data)):
        intermediate_fold_data.append(data[i][:split_index])
        args['testdata'].append(data[i][split_index:])
    # second round of folds
    split_index = int(len(intermediate_fold_data[0]) - (len(intermediate_fold_data[0])*args['validationdata_split']))
    for i in range(len(intermediate_fold_data)):
        args['traindata'].append(intermediate_fold_data[i][:split_index])
        args['validationdata'].append(intermediate_fold_data[i][split_index:])

    print('{} sequences in train data'.format(len(args['traindata'][i])))
    print('{} sequences in test data'.format(len(args['testdata'][i])))
    print('{} sequences in validation data'.format(len(args['validationdata'][i])))

    # split traindata into traindatasplit sets
    if args['traindata_split'] > 1:
        split_traindata = []
        elems_per_split = int(round(len(args['traindata'][0])/args['traindata_split']))
        for split in range(args['traindata_split']):
            split_traindata_array = []
            for i in range(len(args['traindata'])):
                split_traindata_array.append(args['traindata'][i][split*elems_per_split:(split + 1)*elems_per_split])
            split_traindata.append(split_traindata_array)
        args['traindata'] = split_traindata[args['traindata_index']]
        print('{} sequences in split train data'.format(len(args['traindata'][i])))

    # duplicate traindata
    if args['traindata_duplicate'] > 0:
        print('{} elements before duplicating'.format(len(args['traindata'][i])))
        args['traindata'] = regularization.DuplicateData(args['traindata_duplicate'],args['traindata'])
        print('{} elements after duplicating'.format(len(args['traindata'][i])))

    # bag traindata
    if args['bagging']:
        print('{} elements before bagging with putback {}'.format(len(args['traindata'][i]), args['bagging_putback']))
        args['traindata'] = regularization.BagArray(args['bagging_size'],args['traindata'], args['bagging_putback'])
        print('{} elements after bagging with putback {}'.format(len(args['traindata'][i]), args['bagging_putback']))

    # shuffle traindata
    if args['traindata_shuffle']:
        args['traindata'] =  regularization.ShuffleArray(args['traindata'])
        print('traindata shuffled')

    # generate sentences from training data 
    if args['datageneration_pattern'] == DataGenerationPattern.Fit:
        print('perform full in-memory sentence generation')
        args['train_sentences'] = dataoperations.CreateSentences(args['traindata'])
        args['validation_sentences'] = dataoperations.CreateSentences(args['validationdata'])
        print('train_sentences:', len(args['train_sentences'][0]))
        print('validation_sentences:', len(args['validation_sentences'][0]))
    elif args['datageneration_pattern'] == DataGenerationPattern.Generator:
        isTensorflow = args['processor'] == Processor.TPU
        args['train_generator'] = generator.GenerateGenerator(isTensorflow, args['traindata'],args)
        args['validation_generator'] = generator.GenerateGenerator(isTensorflow, args['validationdata'],args,shuffle=False)
        print('created sentence generators')
    else:
        raise ValueError("unknown value for datageneration_pattern")
    return args

def __Train_Model(args):
    # start building input matrix
    if args['datageneration_pattern'] == DataGenerationPattern.Fit:
        print('Vectorization...')
        train_matrices = args['datadefinition'].CreateMatrices(args['train_sentences'],args)
        validation_matrices = args['datadefinition'].CreateMatrices(args['validation_sentences'],args)
        x_train = train_matrices['X']
        y_train = train_matrices['y_t']
        x_val = validation_matrices['X']
        y_val = validation_matrices['y_t']
    elif args['datageneration_pattern'] == DataGenerationPattern.Generator:
        print('Generators detected: Vectorization will be performed during runtime')
    else:
        raise ValueError("unknown value for datageneration_pattern")

    # build the model: 
    print('Build model...')
    model = models.CreateModel(args)
    callbacks = models.CreateCallbacks(args)

    model.summary()
    verbositylevel = 2
    if args['verbose']:
        verbositylevel = 1

    # use tpu calls if tpu is defined
    if(args['processor'] == Processor.TPU):
        #do stuff
        import tensorflow as tf
        model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(args['TPU_WORKER'])))
        args['batch_size'] = args['batch_size'] * 8 #give each tpu batch_size elements    
        print('tpu detected: adjusting batch_size to ',  args['batch_size'])    
    if(args['datageneration_pattern'] == DataGenerationPattern.Fit):
        model.fit(x_train, {'time_output':y_train}, validation_data=(x_val,y_val), verbose=verbositylevel, callbacks=callbacks, shuffle=True, batch_size=args['batch_size'], epochs=args['max_epochs'])
    elif(args['datageneration_pattern'] == DataGenerationPattern.Generator):
        model.fit_generator(generator=args['train_generator'],
            validation_data=args['validation_generator'],
            verbose=verbositylevel, 
            callbacks=callbacks, shuffle=True, epochs=args['max_epochs'])
    else:
        raise ValueError("no training data found")        
    return model

def __Evaluate_Model(args,model = None):
    #prediction:
    print('Load model for predictions...') 

    # save model file if not exists     
    if os.path.exists('{}-model.h5'.format(args['running'])) == False:
        model.save('{}-model.h5'.format(args['running']))  
        print('Model file does not exist, saving...')
    
    if args['save_model'] == False and model is not None:
        # cannot load model because it's not saved, but it is supplied  
        if args['processor'] == Processor.TPU:
            args['processor'] = Processor.CPU # make predictions on a cpu based model
            model.save_weights('{}-modelweights.h5'.format(args['running']))
            import tensorflow as tf # if it runs on tpu: tensorflow
            tf.keras.backend.clear_session()
            model = models.CreateModel(args)
            model.load_weights('{}-modelweights.h5'.format(args['running']))
            print('tpu detected: Model saved with last weights and converted to cpu model for cpu inference')
        else: 
            print('Model loaded from memory')
    elif args['save_model'] == True:
        # model loaded from file, by recreating the model and loading weights
        model = models.CreateModel(args)
        model.load_weights('{}-model.h5'.format(args['running']))
        print('Model loaded from checkpoint')
    else:
        raise ValueError("no model for predictions supplied / found")        
    model.summary()

    #evaluate
    args['datadefinition'].MakePredictions(model,args)    
    configuration.Clean_Session()
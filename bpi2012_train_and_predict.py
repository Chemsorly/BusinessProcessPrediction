import utility.run
import os
import sys
from utility.enums import DataGenerationPattern, Processor, RnnType

#check for env variable for cpu/gpu environment detection
processorType = os.environ.get("CONDUCTHOR_TYPE")
if processorType == "cpu":
    print("cpu environment detected")
elif processorType == "gpu":
    print("gpu environment detected")
else:
    print("unknown environment detected, defaulting to gpu")
    processorType = "cpu"

#import data definition
import datadefinitions.bpi2012 as datadef
datadef = datadef.BPI2012()

if len(sys.argv) > 1:
    param = float(sys.argv[1])
utility.run.Train_And_Evaluate(
    #data
    eventlog="datasets/bpi2012.csv",  # file to read in
    datadefinition=datadef,          # the data / matrix definitions
    running=0,                       # iterable / suffix
    datageneration_pattern = DataGenerationPattern.Fit, # Fit: uses the classical approach and loads everything into memory; Generator: uses the python generator pattern
    #regularization/dataset manipulation
    bagging=False,                   # perform bagging? 
    bagging_putback=True,            # (if bagging) elements get put back and can be drawn again 
    bagging_size=0.8,                # (if bagging) the split to bag train data
    traindata_split=1,               # split the train data into X pieces; default 1 (no split)
    traindata_split_index=0,         # the index of the split to pick; default 0
    traindata_duplicate=0,           # duplicates part of the train data to (virtually) inflate it; default 0 (no duplicates)
    traindata_shuffle=False,         # shuffle training data
    validationdata_split = 0.2,      # Validation data split from traindata (default 0.2)
    testdata_split = 0.3333,         # test data split from input data. takes data from the bottom of the dataset (default 0.333)
    max_sequencelength=150000,       # maximum allowed sequence length
    #framework/ann specifics
    batch_size=64,                   # batch size (set to 1 for stateful)
    rnntype=RnnType.LSTM,            # type of rnn cell to use: lstm, gru or rnn (vanilla)
    neurons=100,                     # neurons per layer
    dropout=0.1,                     # dropout per layer (not applicable to CuDNN)
    max_epochs = 500,                # maximum amount of epochs to run (uses early_stopping)
    layers=2,                        # layers for the rnn
    gradientclipvalue=3,             # value to clip the gradient to if it exceeds it
    patience_earlystopping=20,       # patience for early stopping
    patience_reducelr=20,            # patience for lr reduction
    processor=Processor.GPU,         # processor, cpu, gpu or tpu (gpu uses CUDNN based algorithms for lstm and gru if cudnn is set to true)
    cudnn=True,                      # (if GPU) utilizes special nVidia-CuDNN LSTM and GRU implementations
    stateful=False,                  # stateful or stateless model; default False
    #debug
    save_model=True,                # saves the model file each checkpoint (set to False for TPU usage)
    tensorboard=False,               # outputs tensorboard compatible eventlog into ./graph folder for visualization
    verbose=False)                   # prints out a lot of progress reports. do NOT use with cluster learning


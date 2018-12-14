Python scripts for Keras and Tensorflow to make business process predictions with Recurrent Neural Networks. It reads in event logs (in csv format), trains a network based on supplied parameters and outputs the predictions of the test set.

# How to use
1. define a data definition in the folder datadefinitions  
    a data definition describes the inputs data columns of a dataset and how to handle them  
    1.1. implement GetRowStructure  
    1.2. (optional) override CreateMatrices  
    1.3. (optional) override MakePredictions  
2. implement a start-script, that calls the utility.run.Train_And_Evaluate function with all parameters. Undefined parameters will be replaced by default values.  
    see utility/preprocessing for all default values  
    see c2k_train_and_predict.py for example implementation  
3. run the script. the result will be a trained model (\*.h5) and the predictions of the test set (\*.csv)

# Editable parameters
* datageneration_pattern (enum): defines how the data is fed during training. use Generator if data does not fit into memory.
* bagging (bool): defines if bootstrap aggregating is used for the training set 
* bagging_size (float): relative amount of samples to pick from the training set
* validationdata_split (float, 0-1): relative amount of samples from the traindata to reserve for validation
* testdata_split (float, 0-1): relative amount of samples from the dataset to reserve for test
* max_sequencelength (int): upper bound for maximum length of sequences. longer sequences are dicarded.
* batch_size (int): amount of samples to train simultaneously
* neurons (int): amount of hidden units in each layer
* layers (int): amount of hidden layers
* rnntype (enum): type of RNN units used in hidden layers (RNN,LSTM,GRU)
* bidirectional (bool): use bi-directional hidden layers
* processor (enum): processor to train on (CPU,GPU,TPU)
* cudnn (bool): use CuDNN RNN implementations (GPU only)
* dropout (float, 0-1): dropout (CuDNN does not support dropout yet)
* max_epochs (int): maximum amounts of epoch to train on. usually never reached because of early stopping
* learningrate (float): learning rate
* patience_earlystopping (int): patience for early stopping callback
* patience_reducelr (int): patience for reducelr callback
* gradientclipvalue (float): gradient clipping value
* verbose (bool): output extra information
* tensorboard (bool): add tensorboard callback to output tensorboard data
* save_model (bool): save the intermediate model during checkpoints

# Requirements
* python 3.6
* unicodecsv
* tensorflow 1.9.0
* keras 2.2.4
* (gpu) nvidia graphic driver >=384
* (gpu) Cuda 9
* (gpu) LibCudnn7
* (tpu) [Google Colab](https://colab.research.google.com/)

# Docker images
You can find ready-to-use docker images with all dependencies on [DockerHub](https://cloud.docker.com/repository/docker/chemsorly/keras-tensorflow). Using the GPU variant requires a working installation of [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker).

# References
This code has been used in
* Andreas Metzger, Adrian Neubauer: Considering Non-sequential Control Flows for Process Prediction with Recurrent Neural Networks. SEAA 2018: 268-272 [doi](https://doi.org/10.1109/SEAA.2018.00051) [Repository](https://github.com/Chemsorly/BusinessProcessOutcomePrediction)

# Credits
This code was initially developed during my master thesis and extended during the TransformingTransports research project, which received funding from the EU’s Horizon 2020 R&I programme under grant 731932.  
The Cargo 2000 Freight Tracking and Tracing Data Set is available at [UCL](https://archive.ics.uci.edu/ml/datasets/Cargo+2000+Freight+Tracking+and+Tracing) [(Citation)](http://dx.doi.org/10.1109/TSMC.2014.2347265)  
The BPI Challenge 2012 data set is available at [4tu](https://data.4tu.nl/repository/uuid:3926db30-f712-4394-aebc-75976070e91f)  
The BPI Challenge 2017 data set is available at [4tu](https://data.4tu.nl/repository/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b)  
The BPI Challenge 2018 data set is available at [4tu](https://data.4tu.nl/repository/uuid:3301445f-95e8-4ff0-98a4-901f1f204972)  
Credits go to [verenich](https://github.com/verenich) whose [work](https://github.com/verenich/ProcessSequencePrediction) was used as base for this project [(doi)](https://doi.org/10.1007/978-3-319-59536-8_30).
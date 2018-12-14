from utility.enums import Processor, RnnType, DataGenerationPattern

def Parse_Args(**kwargs):
    outArgs = {}

    # default values if not defined
    outArgs['eventlog'] = ""
    outArgs['running'] = 0
    outArgs['bagging'] = True
    outArgs['bagging_putback'] = True
    outArgs['bagging_size'] = 0.8
    outArgs['traindata_split'] = 1
    outArgs['traindata_index'] = 0
    outArgs['traindata_duplicate'] = 0
    outArgs['traindata_shuffle'] = False
    outArgs['validationdata_split'] = 0.2
    outArgs['testdata_split'] = 0.3333
    outArgs['max_sequencelength'] = 3000
    outArgs['batch_size'] = 16
    outArgs['neurons'] = 100
    outArgs['layers'] = 2
    outArgs['dropout'] = 0.1
    outArgs['max_epochs'] = 500
    outArgs['learningrate'] = 0.002
    outArgs['patience_earlystopping'] = 40
    outArgs['patience_reducelr'] = 10
    outArgs['gradientclipvalue'] = 3
    outArgs['processor'] = Processor.CPU
    outArgs['verbose'] = False
    outArgs['rnntype'] = RnnType.LSTM
    outArgs['bidirectional'] = False
    outArgs['cudnn'] = False
    outArgs['tensorboard'] = False
    outArgs['save_model'] = False
    outArgs['datageneration_pattern'] = DataGenerationPattern.Fit

    # copy in data
    for key,value in kwargs.items():
        outArgs[key] = value
        print('{} defined: {}'.format(key,value)) 

    # default values for specific switches
    if outArgs['processor'] == Processor.TPU:
        outArgs['save_model'] = False
        print('processor = tpu; setting save_model to false, because model_checkpoint slows training down for tpu due to weight copying')

    return outArgs
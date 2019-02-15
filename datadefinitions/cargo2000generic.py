import numpy as np
import csv
import sys
from utility.enums import DataType as dt
from utility.enums import DataClass as dc
from utility.enums import FeatureType as ft
from utility.enums import DataGenerationPattern, Processor, RnnType
from datadefinitions.generic import GenericDatadefinition

class Cargo2000(GenericDatadefinition):
    def GetDataset(self):
        return "datasets/cargo2000.csv"

    def GetRowstructure(self):
        rowstructure = []
        #rowstructure contains the following parameters: (input data type, column id, class of data, type of feature, weight of feature)
        rowstructure.append({'datatype':dt.String, 'columnindex':1 , 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot event id
        #rowstructure.append((dt.String, 10, dc.Onehot, ft.Train, 1.0)) #onehot airport code
        rowstructure.append({'datatype':dt.Float, 'columnindex':3, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #calculated time since start
        rowstructure.append({'datatype':dt.Float, 'columnindex':4, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #timestamp
        rowstructure.append({'datatype':dt.Float, 'columnindex':2, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #duration
        rowstructure.append({'datatype':dt.Float, 'columnindex':5, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #planned duration
        rowstructure.append({'datatype':dt.Float, 'columnindex':6, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #planned timestamp    
        rowstructure.append({'datatype':dt.Float, 'columnindex':8, 'dataclass':dc.Numeric, 'featuretype':ft.Target, 'featureweight':1.0}) #end timestamp
        rowstructure.append({'datatype':dt.Float, 'columnindex':9, 'dataclass':dc.Numeric, 'featuretype':ft.none, 'featureweight':1.0}) #planned end timestamp
        rowstructure.append({'datatype':dt.Int, 'columnindex':7, 'dataclass':dc.none, 'featuretype':ft.none, 'featureweight':1.0}) #processid
        return rowstructure    

    def MakePredictions(self,model,args):
        print('Make predictions...')
        with open('{}-results.csv'.format(args['running']), 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["sequenceid","sequencelength","prefix","completion","prediction","gt_prediction","gt_planned","gt_instance","prefix_activities","suffix_activities"])
            sequenceid = 0
            print('sequences: {}'.format(len(args['testdata'][0])))    
            for i in range(len(args['testdata'][0])):
                sequencelength = len(args['testdata'][0][i]) - 1 #minus eol character
                ground_truth = args['testdata'][6][i][0] + args['offsets'][6] #undo offset
                ground_truth_plannedtimestamp = args['testdata'][7][i][0] + args['offsets'][7] #undo offset
                ground_truth_processid = args['testdata'][8][i][0]
                for prefix_size in range(1,sequencelength):   
                    cropped_data = []
                    for a in range(len(args['testdata'])):
                        cropped_data.append(args['testdata'][a][i][:prefix_size])  
                    prefix_activities = args['testdata'][0][i][:prefix_size]
                    suffix_activities = args['testdata'][0][i][prefix_size:]
                    if '!' in prefix_activities:
                        break # make no prediction for this case, since this case has ended already 

                    # predict
                    y = model.predict(self.__EncodePrediction(cropped_data, args['divisors'], args['indices'],args['feature_weights'], args['num_features'], args['catvectorlen'], args['maxlen']), verbose=0)
                    y_t = y[0][0]
                    y_t = (y_t * args['divisors'][6]) + args['offsets'][6] #undo offset and multiply by divisor to un-normalize
                    prediction = y_t

                    #output stuff (sequence, prefix)
                    output = []
                    output.append(sequenceid)
                    output.append(sequencelength)
                    output.append(prefix_size)
                    output.append(prefix_size / sequencelength)
                    output.append(prediction)                    
                    output.append(ground_truth)
                    output.append(ground_truth_plannedtimestamp)
                    output.append(ground_truth_processid)
                    output.append(' '.join(prefix_activities))
                    output.append(' '.join(suffix_activities))
                    spamwriter.writerow(output) 

                    # out if an prediction was requested for a productive system
                    #if productive_prediction:
                    #    print('PredictedBuffer={}'.format(y_t)) 

                sequenceid += 1
                if args['verbose']:
                    print("finished sequence {} of {}".format(sequenceid,len(args['testdata'][0])))
                #end sequence loop
        print('finished prediction')

    def __EncodePrediction(self, sentence, divisors, indices, weights,  num_features, catvectorlen, maxlen):
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen-len(sentence[0])
        for i in range(len(sentence[0])):
            # set oh vector
            #catvectorpad = 0
            for c in indices["chars"][0]:
                if c == sentence[0][i]:
                    X[0, i+leftpad, indices["chars_indices"][0][c]] = weights[0]
            #catvectorpad += len(indices["chars_indices"][0])
            #for c in indices["chars"][1]:
            #    if c == sentence[1][i]:
            #        X[0, i+leftpad, catvectorpad + indices["chars_indices"][1][c]] = (1 * weights[1])
            X[0, i+leftpad, catvectorlen] = i + 1 #index
            X[0, i+leftpad, catvectorlen+1] = ((sentence[1][i]/divisors[1]) * weights[1]) #time since last event
            X[0, i+leftpad, catvectorlen+2] = ((sentence[2][i]/divisors[2]) * weights[2]) #timestamp
            X[0, i+leftpad, catvectorlen+3] = ((sentence[3][i]/divisors[3]) * weights[3]) #duration
            X[0, i+leftpad, catvectorlen+4] = ((sentence[4][i]/divisors[4]) * weights[4]) #timestamp
            X[0, i+leftpad, catvectorlen+5] = ((sentence[5][i]/divisors[5]) * weights[5]) #duration
        return X
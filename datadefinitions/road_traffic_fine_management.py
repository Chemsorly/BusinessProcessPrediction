import numpy as np
import csv
import sys
from utility.enums import DataType as dt
from utility.enums import DataClass as dc
from utility.enums import FeatureType as ft
from utility.enums import DataGenerationPattern, Processor, RnnType
from datadefinitions.generic import GenericDatadefinition

class RoadTrafficFine(GenericDatadefinition):
    def GetDataset(self):
        return "datasets/road_traffic_fine_management.csv"

    def GetRowstructure(self):
        rowstructure = []
        #rowstructure contains the following parameters: (input data type, column id, class of data, type of feature, weight of feature)
        rowstructure.append({'datatype':dt.String, 'columnindex':2 , 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot event label
        rowstructure.append({'datatype':dt.String, 'columnindex':4 , 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot resource
        rowstructure.append({'datatype':dt.String, 'columnindex':5 , 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot article
        rowstructure.append({'datatype':dt.String, 'columnindex':9 , 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot vehicleclass

        rowstructure.append({'datatype':dt.Float, 'columnindex':3, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #timestamp
        rowstructure.append({'datatype':dt.Float, 'columnindex':6, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #expenses
        rowstructure.append({'datatype':dt.Float, 'columnindex':7, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #fined
        rowstructure.append({'datatype':dt.Float, 'columnindex':8, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #payment
        rowstructure.append({'datatype':dt.Float, 'columnindex':10, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #points

        rowstructure.append({'datatype':dt.String, 'columnindex':11, 'dataclass':dc.Onehot, 'featuretype':ft.Target, 'featureweight':1.0}) #violation t/f
        rowstructure.append({'datatype':dt.String, 'columnindex':12, 'dataclass':dc.none, 'featuretype':ft.none, 'featureweight':1.0}) #processid
        return rowstructure

    def CreateMatrices(self,sentences,args):
        divisors = args['divisors']
        indices = args['indices']
        weights = args['feature_weights']
        num_features = args['num_features']
        catvectorlen = args['catvectorlen']
        maxlen = args['maxlen']
        verbose = args['verbose']

        X = np.zeros((len(sentences[0]), maxlen, num_features), dtype=np.float32)
        y_t = np.zeros((len(sentences[0]),1), dtype=np.float32)

        for i in range (len(sentences[0])):
            leftpad = maxlen-len(sentences[0][i])
            for j in range (len(sentences[0][i])):   
                catvectorpad = 0     
                # set oh vector
                for c in indices["chars"][0]:
                    if c == sentences[0][i][j]:
                        X[i, j+leftpad, indices["chars_indices"][0][c]] = weights[0]
                catvectorpad += len(indices["chars_indices"][0])
                for c in indices["chars"][1]:
                    if c == sentences[1][i][j]:
                        X[i, j+leftpad, catvectorpad + indices["chars_indices"][1][c]] = (1 * weights[1])
                catvectorpad += len(indices["chars_indices"][1])
                for c in indices["chars"][2]:
                    if c == sentences[2][i][j]:
                        X[i, j+leftpad, catvectorpad + indices["chars_indices"][2][c]] = (1 * weights[2])
                catvectorpad += len(indices["chars_indices"][2])
                for c in indices["chars"][3]:
                    if c == sentences[3][i][j]:
                        X[i, j+leftpad, catvectorpad + indices["chars_indices"][3][c]] = (1 * weights[3])

                # set time vector
                X[i, j+leftpad, catvectorlen] = j + 1 #index
                X[i, j+leftpad, catvectorlen+1] = ((sentences[4][i][j]/divisors[4]) * weights[4])  #time since last event
                X[i, j+leftpad, catvectorlen+2] = ((sentences[5][i][j]/divisors[5]) * weights[5])  #timestamp
                X[i, j+leftpad, catvectorlen+3] = ((sentences[6][i][j]/divisors[6]) * weights[6])  #duration
                X[i, j+leftpad, catvectorlen+4] = ((sentences[7][i][j]/divisors[7]) * weights[7])  #planned duration
                X[i, j+leftpad, catvectorlen+5] = ((sentences[8][i][j]/divisors[8]) * weights[8])  #planned timestamp

            if sentences[9][i][0] == "False":
                y_value = 0
            else:
                y_value = 1
            y_t[i,0] = y_value
            if verbose and args['datageneration_pattern'] != DataGenerationPattern.Generator:
                sys.stdout.write("vectorized sequence {0} of {1}\r".format(i,len(sentences[0])))
                sys.stdout.flush()
        return {'X': X, 'y_t': y_t}

    def MakePredictions(self,model,args):
        print('Make predictions...')
        with open(args['testresultsfilename'], 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["sequenceid","sequencelength","prefix","completion","prediction","gt_prediction","gt_planned","gt_instance","prefix_activities","suffix_activities"])
            sequenceid = 0
            print('sequences: {}'.format(len(args['testdata'][0])))    
            for i in range(len(args['testdata'][0])):
                sequencelength = len(args['testdata'][0][i]) - 1 #minus eol character
                ground_truth = args['testdata'][9][i][0]
                ground_truth_processid = args['testdata'][10][i][0]
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
                    prediction = y[0][0]

                    #output stuff (sequence, prefix)
                    output = []
                    output.append(sequenceid)
                    output.append(sequencelength)
                    output.append(prefix_size)
                    output.append(prefix_size / sequencelength)
                    output.append(prediction)                    
                    output.append(ground_truth)
                    output.append(0.5)
                    output.append(ground_truth_processid)
                    output.append(' ')
                    output.append(' ')
                    #output.append(' '.join(prefix_activities))
                    #output.append(' '.join(suffix_activities))
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
            catvectorpad = 0
            for c in indices["chars"][0]:
                if c == sentence[0][i]:
                    X[0, i+leftpad, indices["chars_indices"][0][c]] = weights[0]
            catvectorpad += len(indices["chars_indices"][0])
            for c in indices["chars"][1]:
                if c == sentence[1][i]:
                    X[0, i+leftpad, catvectorpad + indices["chars_indices"][1][c]] = (1 * weights[1])
            catvectorpad += len(indices["chars_indices"][1])
            for c in indices["chars"][2]:
                if c == sentence[2][i]:
                    X[0, i+leftpad, catvectorpad + indices["chars_indices"][2][c]] = (1 * weights[2])
            catvectorpad += len(indices["chars_indices"][2])
            for c in indices["chars"][3]:
                if c == sentence[3][i]:
                    X[0, i+leftpad, catvectorpad + indices["chars_indices"][3][c]] = (1 * weights[3])
            X[0, i+leftpad, catvectorlen] = i + 1 #index
            X[0, i+leftpad, catvectorlen+1] = ((sentence[4][i]/divisors[4]) * weights[4]) #time since last event
            X[0, i+leftpad, catvectorlen+2] = ((sentence[5][i]/divisors[5]) * weights[5]) #timestamp
            X[0, i+leftpad, catvectorlen+3] = ((sentence[6][i]/divisors[6]) * weights[6]) #duration
            X[0, i+leftpad, catvectorlen+4] = ((sentence[7][i]/divisors[7]) * weights[7]) #timestamp
            X[0, i+leftpad, catvectorlen+5] = ((sentence[8][i]/divisors[8]) * weights[8]) #duration
        return X
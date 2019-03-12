import numpy as np
import csv
import sys
from utility.enums import DataType as dt
from utility.enums import DataClass as dc
from utility.enums import FeatureType as ft
from utility.enums import DataGenerationPattern, Processor, RnnType
from datadefinitions.generic import GenericDatadefinition

class Sepsis(GenericDatadefinition):
    def GetDataset(self):
        return "datasets/sepsis_cases.csv"

    def GetRowstructure(self):
        rowstructure = []
        #rowstructure contains the following parameters: (input data type, column id, class of data, type of feature, weight of feature)
        rowstructure.append({'datatype':dt.String, 'columnindex':2, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) # (0) onehot event label
        rowstructure.append({'datatype':dt.String, 'columnindex':5, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot orggroup
        rowstructure.append({'datatype':dt.String, 'columnindex':6, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot InfectionSuspected
        rowstructure.append({'datatype':dt.String, 'columnindex':7, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':8, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':9, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':10, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':11, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':12, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':13, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':14, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':15, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':16, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':17, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':18, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':19, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':20, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':21, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':22, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':23, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':24, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':25, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':26, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':27, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) #onehot ...
        rowstructure.append({'datatype':dt.String, 'columnindex':28, 'dataclass':dc.Onehot, 'featuretype':ft.Train, 'featureweight':1.0}) # (24) onehot ...

        rowstructure.append({'datatype':dt.Float, 'columnindex':29, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #age
        rowstructure.append({'datatype':dt.Float, 'columnindex':30, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #crp
        rowstructure.append({'datatype':dt.Float, 'columnindex':31, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #leucocytes
        rowstructure.append({'datatype':dt.Float, 'columnindex':32, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #laticacid
        rowstructure.append({'datatype':dt.Float, 'columnindex':3, 'dataclass':dc.Numeric, 'featuretype':ft.Train, 'featureweight':1.0}) #timestamp

        rowstructure.append({'datatype':dt.String, 'columnindex':4, 'dataclass':dc.Onehot, 'featuretype':ft.Target, 'featureweight':1.0}) #violation t/f
        rowstructure.append({'datatype':dt.String, 'columnindex':33, 'dataclass':dc.none, 'featuretype':ft.none, 'featureweight':1.0}) #processid
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
                for index in range(0,25): #0-24
                    for c in indices["chars"][index]:
                        if c == sentences[index][i][j]:
                            X[i, j+leftpad, indices["chars_indices"][index][c]] = weights[index]
                    catvectorpad += len(indices["chars_indices"][index])                

                # set time vector
                X[i, j+leftpad, catvectorlen] = j + 1 #index
                X[i, j+leftpad, catvectorlen+1] = ((sentences[25][i][j]/divisors[25]) * weights[25])  #time since last event
                X[i, j+leftpad, catvectorlen+2] = ((sentences[26][i][j]/divisors[26]) * weights[26])  #timestamp
                X[i, j+leftpad, catvectorlen+3] = ((sentences[27][i][j]/divisors[27]) * weights[27])  #duration
                X[i, j+leftpad, catvectorlen+4] = ((sentences[28][i][j]/divisors[28]) * weights[28])  #planned duration
                X[i, j+leftpad, catvectorlen+5] = ((sentences[29][i][j]/divisors[29]) * weights[29])  #planned timestamp

            if sentences[30][i][0] == "False":
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
                ground_truth = args['testdata'][30][i][0]
                ground_truth_processid = args['testdata'][31][i][0]
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
            for index in range(0,25):
                for c in indices["chars"][index]:
                    if c == sentence[index][i]:
                        X[0, i+leftpad, indices["chars_indices"][index][c]] = weights[index]
                catvectorpad += len(indices["chars_indices"][index])
            X[0, i+leftpad, catvectorlen] = i + 1 #index
            X[0, i+leftpad, catvectorlen+1] = ((sentence[25][i]/divisors[25]) * weights[25]) #time since last event
            X[0, i+leftpad, catvectorlen+2] = ((sentence[26][i]/divisors[26]) * weights[26]) #timestamp
            X[0, i+leftpad, catvectorlen+3] = ((sentence[27][i]/divisors[27]) * weights[27]) #duration
            X[0, i+leftpad, catvectorlen+4] = ((sentence[28][i]/divisors[28]) * weights[28]) #timestamp
            X[0, i+leftpad, catvectorlen+5] = ((sentence[29][i]/divisors[29]) * weights[29]) #duration
        return X
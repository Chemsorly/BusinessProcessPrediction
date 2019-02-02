import numpy as np
import csv
import sys
from utility.enums import DataType as dt
from utility.enums import DataClass as dc
from utility.enums import FeatureType as ft
from utility.enums import DataGenerationPattern, Processor, RnnType
from datadefinitions.generic import GenericDatadefinition

class BPI2012(GenericDatadefinition):
    def GetDataset(self):
        return "datasets/bpi2012.csv"

    def GetRowstructure(self):
        rowstructure = []
        rowstructure.append((dt.String, 2 , dc.Onehot, ft.Train, 1.0)) #event label (onehot) 
        rowstructure.append((dt.String, 5, dc.Onehot, ft.Train, 1.0 )) #resource   
        rowstructure.append((dt.Float, 3, dc.Numeric, ft.Train, 1.0 )) #duration (always 0)
        rowstructure.append((dt.Float, 4, dc.Numeric, ft.Train, 1.0 )) #timestamp    
    
        rowstructure.append((dt.String, 6, dc.Onehot, ft.Target )) #t/f
        rowstructure.append((dt.String, 7, dc.none, ft.none )) #caseid
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
                # set oh vector (event label)
                for c in indices["chars"][0]:
                    if c == sentences[0][i][j]:
                        X[i, j+leftpad, indices["chars_indices"][0][c]] = 1
                catvectorpad += len(indices["chars_indices"][0])
                # set oh vector (resources)
                for c in indices["chars"][1]:
                    if c == sentences[1][i][j]:
                        X[i, j+leftpad, catvectorpad + indices["chars_indices"][1][c]] = 1
                # set time vector
                X[i, j+leftpad, catvectorlen] = j + 1 #index
                X[i, j+leftpad, catvectorlen+1] = sentences[2][i][j]/divisors[2]
                X[i, j+leftpad, catvectorlen+2] = sentences[3][i][j]/divisors[3] 

            if sentences[4][i][0] == "False":
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
        with open('{}-results.csv'.format(args['running']), 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["sequenceid","sequencelength","prefix","completion","prediction","gt_prediction","gt_planned","gt_instance","prefix_activities","suffix_activities"])
            sequenceid = 0
            print('sequences: {}'.format(len(args['testdata'][0])))    
            for i in range(len(args['testdata'][0])):
                sequencelength = len(args['testdata'][0][i]) - 1 #minus eol character
                ground_truth = args['testdata'][4][i][0]
                ground_truth_processid = args['testdata'][5][i][0]
                for prefix_size in range(1,sequencelength):   
                    cropped_data = []
                    for a in range(len(args['testdata'])):
                        cropped_data.append(args['testdata'][a][i][:prefix_size])  
                    prefix_activities = args['testdata'][0][i][:prefix_size]
                    suffix_activities = args['testdata'][0][i][prefix_size:]

                    # predict
                    y = model.predict(self.__EncodePrediction(cropped_data, args['divisors'], args['indices'], args['num_features'], args['catvectorlen'], args['maxlen']), verbose=0)
                    y_t = y[0][0]
                    prediction = y_t

                    #output stuff (sequence, prefix)
                    output = []
                    output.append(sequenceid)
                    output.append(sequencelength)
                    output.append(prefix_size)
                    output.append(prefix_size / sequencelength)
                    output.append(prediction)                    
                    output.append(ground_truth)
                    output.append(0.5) # binary prediction with [0...1]
                    output.append(ground_truth_processid)
                    output.append(' '.join(prefix_activities))
                    output.append(' '.join(suffix_activities))
                    spamwriter.writerow(output)    

                sequenceid += 1
                if args['verbose']:
                    print("finished sequence {} of {}".format(sequenceid,len(args['testdata'][0])))
                #end sequence loop
        print('finished prediction')

    def __EncodePrediction(self,sentence, divisors, indices,  num_features, catvectorlen, maxlen):
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen-len(sentence[0])
        for i in range(len(sentence[0])):
            catvectorpad = 0 
            # set oh vector
            for c in indices["chars"][0]:
                if c == sentence[0][i]:
                    X[0, i+leftpad, indices["chars_indices"][0][c]] = 1
            catvectorpad += len(indices["chars_indices"][0])
            # set ml vector
            for c in indices["chars"][1]:
                if c == sentence[1][i]:
                    X[0, i+leftpad, catvectorpad + indices["chars_indices"][1][c]] = 1
            X[0, i+leftpad, catvectorlen] = i + 1 #index
            X[0, i+leftpad, catvectorlen+1] = sentence[2][i]/divisors[2] 
            X[0, i+leftpad, catvectorlen+2] = sentence[3][i]/divisors[3]
        return X
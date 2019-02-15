import numpy as np
import csv
import sys
from abc import ABC, abstractmethod
from utility.enums import DataType as dt
from utility.enums import DataClass as dc
from utility.enums import FeatureType as ft

class GenericDatadefinition(ABC): 
    """ generic implementation for the functions CreateMatrices and MakePredictions. needs to override GetDataset and GetRowstructure """

    @abstractmethod
    def GetDataset(self):
        raise NotImplementedError("not implemented")
     
    @abstractmethod
    def GetRowstructure(self):        
        raise NotImplementedError("not implemented")

    def CreateMatrices(self,sentences,args):
        divisors = args['divisors']
        indices = args['indices']
        weights = args['feature_weights']
        num_features = args['num_features']
        rowstructure = args['rowstructure']
        maxlen = args['maxlen']
        verbose = args['verbose']

        X = np.zeros((len(sentences[0]), maxlen, num_features), dtype=np.float32)
        y_t = np.zeros((len(sentences[0]),1), dtype=np.float32)

        for i in range(len(sentences[0])):
            leftpad = maxlen-len(sentences[0][i])
            
            # train matrix
            self.__EncodeMatrix(X,i,leftpad,rowstructure,sentences,divisors,indices,weights)
            
            # target matrix
            for k, struc in enumerate(rowstructure): 
                # target variables
                if struc['featuretype'] == ft.Target:
                    # TODO: more than 1 target
                    y_t[i,0] = sentences[k][i][0]/divisors[k] #already offsetted

            if verbose and args['datageneration_pattern'] != DataGenerationPattern.Generator:
                sys.stdout.write("vectorized sequence {0} of {1}\r".format(i,len(sentences[0])))                    
                sys.stdout.flush()
        return {'X': X, 'y_t': y_t}

    def MakePredictions(self,model,args):
        #TODO
        print('Make predictions...')
        with open('{}-results.csv'.format(args['running']), 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["sequenceid","sequencelength","prefix","completion","prediction","gt_prediction","gt_planned","gt_instance","prefix_activities","suffix_activities"])
            sequenceid = 0
            print('sequences: {}'.format(len(args['testdata'][0])))    
            for i in range(len(args['testdata'][0])):
                sequencelength = len(args['testdata'][0][i]) - 1 #minus eol character
                ground_truth = args['testdata'][6][i][0] + args['offsets'][6] #undo offset
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
                    y = model.predict(__EncodePrediction(cropped_data, args['rowstructure'], args['divisors'], args['indices'],args['feature_weights'], args['num_features'], args['catvectorlen'], args['maxlen']), verbose=0)
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
                    output.append("")
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

    def __EncodePrediction(self, sentence, rowstructure, divisors, indices, weights,  num_features, catvectorlen, maxlen):
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        leftpad = maxlen-len(sentence[0])
        self.__EncodeMatrix(X, 0, leftpad, rowstructure, sentence, divisors, indices, weights)
        return X
    
    def __EncodeMatrix(self, X, index, leftpad, rowstructure, sentences, divisors, indices, weights):
        for j in range (len(sentences[0][index])):
            catvectorpad = 0  
            # train matrix
            for k, struc in enumerate(rowstructure):                         
                if struc['featuretype'] == ft.Train:
                    # manage strings/labels
                    if struc['datatype'] == dt.String:
                        if struc['dataclass'] == dc.Onehot:
                            for c in indices["chars"][k]:
                                if c == sentences[k][index][j]:
                                    X[index, j+leftpad, catvectorpad + indices["chars_indices"][k][c]] = weights[k]
                            catvectorpad += len(indices["chars_indices"][k])
                        elif struc['dataclass'] == dc.Multilabel:
                            for c in indices["unique_chars"][k]:
                                if c in sentences[k][index][j]:
                                    X[index, j+leftpad, catvectorpad + indices["unique_chars_indices"][k][c]] = weights[k]
                            catvectorpad += len(indices["unique_chars_indices"][k])
                        else:
                            raise NotImplementedError("string dataclass not implemented")
                    # manage numerics
                    elif struc['datatype'] == dt.Float:
                        if struc['dataclass'] == dc.Numeric:
                            X[index, j+leftpad, catvectorpad] = ((sentences[k][index][j]/divisors[k]) * weights[k]) 
                            catvectorpad += 1
                        elif struc['dataclass'] == dc.Periodic:
                            raise NotImplementedError("periodic dataclass not implemented")
                        else:
                            raise NotImplementedError("numeric dataclass not implemented")
                    # special case: no dataclasses for int; treat as numeric
                    elif struc['datatype'] == dt.Int:                                
                        X[index, j+leftpad, catvectorpad] = ((sentences[k][index][j]/divisors[k]) * weights[k]) 
                        catvectorpad += 1
                    else:
                        raise NotImplementedError("datatype not implemented")
            X[index, j+leftpad, catvectorpad] = j + 1 #index    


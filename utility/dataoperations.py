import csv
import unicodecsv
import numpy as np
import copy
import sys
from .enums import DataType as dt
from .enums import DataClass as dc
from .enums import FeatureType as ft

minutes_in_day = 24 * 60
days_in_week = 7

def ReadInData(file, row_structure):
    csvfile = open(file, 'r')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader, None)  # skip the headers
    lastcase = ''
    firstLine = True
    totalrows = 0

    #create return array with size of len(row_structure)
    data = []
    intermediateData = []
    for i in range(len(row_structure)):
        data.append([])
        intermediateData.append([])

    for row in spamreader:
        totalrows = totalrows + 1
        if row[0] != lastcase:
            lastcase = row[0]
            if not firstLine:
                for i in range(len(row_structure)):
                    data[i].append(intermediateData[i])
            for i in range(len(row_structure)):
                intermediateData[i] = []
        for i in range(len(row_structure)):
            #cast by defined type
            if row_structure[i][0] == dt.String:                
                intermediateData[i].append(row[row_structure[i][1]])
            elif row_structure[i][0] == dt.Float:
                intermediateData[i].append(float(row[row_structure[i][1]]))
            elif row_structure[i][0] == dt.Int:
                intermediateData[i].append(int(row[row_structure[i][1]]))
            else:
                raise ValueError("unknown datatype encountered")
        firstLine = False
    # add last case
    for i in range(len(row_structure)):
        data[i].append(intermediateData[i])
    print("read {} rows".format(totalrows))
    return data

def CalculateFeatures(args):
    catvectorlen = 0
    num_features = 1 # default 1 because indexing variable
    args['feature_weights'] = []
    for index, definition in enumerate(args['rowstructure']):
        if definition[3] == ft.Train:
            args['feature_weights'].append(definition[4]) #append feature weight
            if definition[0] == dt.String:
                if definition[2] == dc.Onehot:
                    catvectorlen += len(args['indices']['chars_indices'][index]) # onehot increases vector length by amount of classes (encoded as strings)
                elif definition[2] == dc.Multilabel:
                    catvectorlen += len(args['indices']['unique_chars_indices'][index]) # ml increases vector length by amount of classes (encoded as chars)
            elif definition[0] == dt.Float:
                num_features += 1 # every numeric value increases by 1 (TODO: periodic)
            elif definition[0] == dt.Int:
                num_features += 1 # every numeric value increases by 1 (TODO: periodic)  

    args['catvectorlen'] = catvectorlen
    print('category vectors length:', args['catvectorlen'])
    args['num_features'] = catvectorlen + num_features
    print('num features: {}'.format(args['num_features']))
    return

def AppendEOL(data):
    outdata = []
    for i in range(len(data)):
        if isinstance(data[i][0], float) or isinstance(data[i][0], int):
            outdata.append(list(map(lambda x: x + [0], data[i])))
        else:
            outdata.append(list(map(lambda x: x + ['!'],data[i])))
    return outdata

def TruncateSequences(indata, maxlength):
    outdata = []
    for i in range(len(indata)):
        outdata.append([])
    for i in range(len(indata[0])):
        if(len(indata[0][i]) <= maxlength):
            for j in range(len(indata)):
                outdata[j].append(indata[j][i])
    print("sequences truncated to {}".format(maxlength))
    return outdata

def CreateDivisors(data):
    # get divisors for normalization
    divisors = []
    for i in range(len(data)):
        if isinstance(data[i][0][0], float) or isinstance(data[i][0][0], int):
            divisors.append(np.mean([item for sublist in data[i] for item in sublist]))
        else:
            divisors.append("null")
        print('divisor{0}: {1}'.format(i, divisors[i]))
    return divisors

def CreateOffsets(data):
    offsets = []
    for i in range(len(data)):
        if isinstance(data[i][0][0], float) or isinstance(data[i][0][0], int):
            # normalize to: 0 to x
            offset = np.min([item for sublist in data[i] for item in sublist])
            if offset < 0:
                datatemp = []
                for seq in data[i]:
                    datatempint = []
                    for elem in seq:
                        datatempint.append(elem - offset)
                    datatemp.append(datatempint)
                data[i] = datatemp
                offsets.append(offset)
            else:
                offsets.append(0)
        else:
            offsets.append("null")
        print('offset{0}: {1}'.format(i, offsets[i]))
    return offsets

def CreateDictionaries(data, rowstructure):    
    resultDict = {
        "chars" : [], # corpus
        "unique_chars" : [], # corpus elements from multilabel classification
        "target_chars" : [], # corpus including eol
        "target_unique_chars" : [], # corpus elements from multilabel classification including eol

        "chars_indices" : [], # chars - index mapping
        "unique_chars_indices" : [], # unique chars - index mapping
        "target_chars_indices" : [], # target chars - index mapping
        "target_unique_chars_indices" : [],

        "indices_chars" : [], # index - chars mapping
        "indices_target_chars" : [], # index - target chars mapping
        "indices_unique_chars" : [], # index - unique chars mapping
        "indices_target_unique_chars" : []
    }

    for i in range(len(data)): 
        if isinstance(data[i][0][0], str) and (rowstructure[i][2] == dc.Onehot or rowstructure[i][2] == dc.Multilabel):
            # get chars
            buffer = map(lambda x : set(x),data[i])
            buffer = list(set().union(*buffer))
            buffer.sort()
            resultDict["target_chars"].append(copy.copy(buffer)) # fill target_chars
            buffer.remove('!')
            resultDict["chars"].append(buffer) # fill chars
            print('total chars: {}, target chars: {}'.format(len(resultDict["chars"][i]), len(resultDict["target_chars"][i])))
            print('characters: ', resultDict["chars"][i])

            #get unique chars
            buffer = [l for word in resultDict["chars"][i] for l in word]
            buffer.append('!')
            buffer = list(set(buffer))
            buffer.sort()
            resultDict["target_unique_chars"].append(copy.copy(buffer)) # fill target_unique_chars
            buffer.remove('!')
            resultDict["unique_chars"].append(buffer) # fill unique_chars
            print('unique characters: ', resultDict["unique_chars"][i])

            # map index dictionaries
            resultDict["chars_indices"].append(dict((c, i) for i, c in enumerate(resultDict["chars"][i]))) #dictionary<key,value> with <char, index> where char is unique symbol for activity
            resultDict["unique_chars_indices"].append(dict((c, i) for i, c in enumerate(resultDict["unique_chars"][i])))
            resultDict["target_chars_indices"].append(dict((c, i) for i, c in enumerate(resultDict["target_chars"][i])))
            resultDict["target_unique_chars_indices"].append(dict((c, i) for i, c in enumerate(resultDict["target_unique_chars"][i])))
            resultDict["indices_chars"].append(dict((i, c) for i, c in enumerate(resultDict["chars"][i]))) #dictionary<key,value> with <index, char> where char is unique symbol for activity
            resultDict["indices_unique_chars"].append( dict((i, c) for i, c in enumerate(resultDict["unique_chars"][i])))
            resultDict["indices_target_chars"].append(dict((i, c) for i, c in enumerate(resultDict["target_chars"][i])))
            resultDict["indices_target_unique_chars"].append(dict((i, c) for i, c in enumerate(resultDict["target_unique_chars"][i])))
        else: # append empty dictionaries for non-string types
            resultDict["target_chars"].append([])
            resultDict["chars"].append([])
            resultDict["target_unique_chars"].append([])
            resultDict["unique_chars"].append([])
            resultDict["chars_indices"].append([]) 
            resultDict["unique_chars_indices"].append([])
            resultDict["target_chars_indices"].append([])
            resultDict["target_unique_chars_indices"].append([])
            resultDict["indices_chars"].append([])
            resultDict["indices_unique_chars"].append([])
            resultDict["indices_target_chars"].append([])
            resultDict["indices_target_unique_chars"].append([])
    return resultDict

def CreateSentences(data):
    sentences = []
    for i in range(len(data)):
        buffer = []
        for j in range(len(data[0])):
            for k in range(len(data[0][j])):
                if k == 0: # skip sequence with length = 1
                    continue
                buffer.append(data[i][j][0:k])
        sentences.append(buffer)
    return sentences
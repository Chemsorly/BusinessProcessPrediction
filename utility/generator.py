import numpy as np
import random
from utility.dataoperations import CreateSentences
from utility.regularization import ShuffleArray

#TODO: make tpu compliant
#TODO: shuffle switch

def count_sentences(data):
    datasize = 0
    for i in range(0,len(data[0])):
        datasize += len(data[0][i]) - 1 # remove EOL character
    print("count generator with {} sequences and {} sentences".format(len(data[0]), datasize))
    return datasize

def GenerateGenerator(isTensorflow, data, args,shuffle=True):
    # factory function
    if isTensorflow == True:
        import tensorflow.keras as keras
    else:
        import keras as keras
    generator = __GenerateGenerator(keras.utils.Sequence, data, args, shuffle)
    return generator

def __GenerateGenerator(base, data, args, shuffle):
    class DataGenerator(base):
            'Generates data for Keras'
            def __init__(self, data, args, shuffle):
                'Initialization'
                self.data = data
                self.shuffle = shuffle
                self.datadefinition = args['datadefinition']
                self.args = args
                self.datasize = 0
                self.current_index = 0

                self.buffer = []
                self.outbuffer = []
                self.newbuffer = []

                #self.on_epoch_end()
                #self.__shuffle_data()

                for i in range(len(data)):
                    self.buffer.append([])

                for i in range(0,len(self.data[0])):
                    self.datasize += len(self.data[0][i]) - 1 # remove EOL character
                print("created generator with with {} sequences and {} sentences".format(len(self.data[0]), self.datasize))

                if self.shuffle:
                    self.__shuffle_data()

            def __len__(self):
                'Denotes the number of batches per epoch'
                return int(np.floor(self.datasize / self.args['batch_size']))

            def __getitem__(self, index):
                'Generate one batch of data'
                # fill buffer until threshold reached
                while len(self.buffer[0]) < self.args['batch_size']:
                    sequence = []
                    for i in range(len(self.data)):
                        sequence.append([])
                        sequence[i].append(self.data[i][self.current_index])      
                    sentences = CreateSentences(sequence)
                    self.current_index +=1 
                    if self.current_index >= len(data[0]):
                        self.current_index = 0
                    # put sentences into buffer
                    for i in range(len(self.data)):
                        for j in range(len(sentences[i])):
                            self.buffer[i].append(sentences[i][j])
                
                # if buffer has reached certain size, yield
                self.outbuffer = []
                self.newbuffer = []
                for i in range(len(self.buffer)):
                    self.outbuffer.append([])
                    self.newbuffer.append([])
                    for j in range(len(self.buffer[i])):
                        if j < self.args['batch_size']:
                            self.outbuffer[i].append(self.buffer[i][j])
                        else:
                            self.newbuffer[i].append(self.buffer[i][j])
                # remove used entries by setting a new buffer with remaining entries
                self.buffer = self.newbuffer
                matrix = self.datadefinition.CreateMatrices(self.outbuffer,self.args)

                return matrix['X'], matrix['y_t']        

            def on_epoch_end(self):
                'after each epoch: reset index and shuffle'
                self.current_index = 0
                if self.shuffle == True:
                    self.__shuffle_data()

            def __shuffle_data(self):
                self.data = ShuffleArray(self.data)
    return DataGenerator(data, args, shuffle)

    
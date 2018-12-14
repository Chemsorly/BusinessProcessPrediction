import random
import numpy

def ShuffleArray(array):
    random_state = numpy.random.get_state()
    intermediate = []
    for i in range(len(array)):
        numpy.random.shuffle(array[i])
        shuffledArray = array[i]
        # reset state to before shuffle, to ensure same shuffling result https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
        numpy.random.set_state(random_state) 
        intermediate.append(shuffledArray)
    return intermediate

def BagArray(bagging_size, array, putback):
    intermediate = []
    if putback == False and bagging_size >= 1:
        bagging_size = 1
    indices = list(range(0, int(round(bagging_size * len(array[0])))))
    for i in range(len(array)):
        intermediate.append([])
    if putback == True:
        for i in range(int(round(bagging_size * len(array[0])))):
            num = random.randint(0, len(array[0]) - 1)
            for j in range(len(array)):
                intermediate[j].append(array[j][num])
    elif putback == False:
        narray = ShuffleArray(array)
        for i in indices:
            for j in range(len(narray)):
                intermediate[j].append(narray[j][i])
    return intermediate

def DuplicateData(duplicate_size, array):
    intermediate = []
    # create empty array
    for i in range(len(array)):
        intermediate.append([])
    # copy array
    for i in range(len(array[0])):
        for j in range(len(array)):
            intermediate[j].append(array[j][i])
    # add additional duplicates
    number_of_dupes = int(round(duplicate_size * len(array[0])))
    if number_of_dupes > 0:     
        # iterate through entire set (number_of_dupes > 1)
        for i in range(int(number_of_dupes / len(array[0]))):
            for j in range(len(array[0])):
                for k in range(len(array)):
                    intermediate[k].append(array[k][j])              
        # iterate through part of set (modulo)        
        for i in range(number_of_dupes % len(array[0])): 
            for j in range(len(array)):
                intermediate[j].append(array[j][i])

    return intermediate

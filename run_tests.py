import utility.run
import os
import sys
from utility.enums import DataGenerationPattern, Processor, RnnType

# datasets to test
import datadefinitions.cargo2000 as cargo2000
import datadefinitions.cargo2000generic as cargo2000generic
import datadefinitions.bpi2012 as bpi2012
import datadefinitions.bpi2017 as bpi2017
import datadefinitions.bpi2018 as bpi2018

# test all testsets
for i, test_set in enumerate([cargo2000.Cargo2000(),cargo2000generic.Cargo2000(),bpi2012.BPI2012(),bpi2017.BPI2017(),bpi2018.BPI2018()]):
    # run with low profile and default values to speed up tests
    utility.run.Train_And_Evaluate(
        datageneration_pattern = DataGenerationPattern.Generator,
        datadefinition=test_set,
        running=i,
        bagging=True, 
        bagging_size=0.05, 
        validationdata_split = 0.05, 
        testdata_split = 0.05,  
        max_sequencelength=50, 
        batch_size=64,   
        neurons=10,  
        dropout=0.1,
        max_epochs = 2,
        layers=1,
        save_model=False)           

# test network configurations
cargo2000_set = cargo2000.Cargo2000()
for a, rnntype in enumerate([RnnType.LSTM, RnnType.GRU, RnnType.RNN]):
        for b, bidirectional in enumerate([True, False]):
                for layers in range(1,4):
                        utility.run.Train_And_Evaluate(
                                datadefinition=cargo2000_set,
                                bagging=True, 
                                bagging_size=0.05, 
                                validationdata_split = 0.05, 
                                testdata_split = 0.05,  
                                max_sequencelength=50,   
                                neurons=10,  
                                max_epochs = 2,
                                layers=layers,
                                rnntype=rnntype,
                                bidirectional=bidirectional,
                                save_model=False)  
 

from enum import Enum

class DataType(Enum):
    String = "string"
    Float = "float"
    Int = "int"

class DataClass(Enum):
    Onehot = "onehot"
    Multilabel = "multilabel"
    Numeric = "numeric"
    Periodic = "periodic"
    Integer = "integer"
    none = "none"

class RnnType(Enum):
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"

class Processor(Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"

class FeatureType(Enum):
    Train = "train"
    Target = "target"
    none = "none"

class DataGenerationPattern(Enum):
    Fit = "Fit" # classic approach (aka load all into memory)
    Generator = "Generator" # generator approach (e.g. fit_generator)

#class GeneratorReadingPattern(Enum):
    #Memory = "Memory" # loads entire dataset into memory, creates batches on the fly
    #Filesystem = "Filesystem" # reads processes from filesystem, uses a lookup table to remember start location in file; not yet implemented
    #Database = "Database" # for database shenanigans; not yet implemented
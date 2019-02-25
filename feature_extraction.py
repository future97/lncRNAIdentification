import os
import re
import csv
import random
import time
import numpy as np
from math import floor
from pandas import DataFrame
# from sequence_attributes_kmer import SequenceAttributes
from sequence_attributes_orf import SequenceAttributes

def feature_extract(bacdsfile, balncrnafile, trainingfile, testingfile, fraction = 0.75):
    
    if bacdsfile == balncrnafile:
        print("Is not allowed to input the same list of files for lncRNAs and PCTs.")
        exit(1)
    
    l = SequenceAttributes(input_file = balncrnafile, clazz = 1)
    p = SequenceAttributes(input_file = bacdsfile, clazz = 0)

    
    print("Processing long non-coding RNA fasta file '%s'..." % balncrnafile)
    l.process()
    print("Processing proteing coding transcripts fasta file '%s'..." % bacdsfile)
    p.process()
    size = min([len(l.data), len(p.data)])
    
    longs_data_training, longs_data_testing = section(l.data, size,fraction)
    pcts_data_training, pcts_data_testing = section(p.data, size,fraction)

    training = np.hstack((longs_data_training, pcts_data_training))
    testing = np.hstack((longs_data_testing, pcts_data_testing))
    #输出csv文件
    pdtraining = DataFrame(training)
    pdtraining.to_csv(trainingfile, index=False)
    pdtesting = DataFrame(testing)
    pdtesting.to_csv(testingfile, index=False)
    
#数据分割
def section(data, size, fraction):
    randomdata = range(size)
    randomsize = int(floor(size * fraction))
    idx = random.sample(randomdata, randomsize)
    mask = np.ones(size, np.bool)
    mask[idx] = 0
    return data[idx], data[mask]
    
#程序入口
if __name__ == '__main__':
    time_start=time.time()
    bacdsfile= './data/Homo_sapiens.GRCh38.cds.test.fa'
    balncrnafile= './data/Homo_sapiens.GRCh38.lncrna.test.fa'
    trainingfile= './data/Homo_sapiens_GRCh38_orf_test_training.csv'
    testingfile= './data/Homo_sapiens_GRCh38_orf_test_testing.csv'
    feature_extract(bacdsfile, balncrnafile, trainingfile, testingfile)
    time_end=time.time()
    print('totally cost',time_end-time_start)

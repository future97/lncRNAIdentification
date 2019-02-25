import time
import numpy as np
from Bio import SeqIO

#平衡数据集，输入编码RNA和长非编码RNA文件，输出平衡后的编码RNA和长非编码RNA文件
def process(cdsfile, lncrnafile, outputcdsfile, outputlncrnafile):
    datacds = []
    datalncrna = []
    with open(cdsfile, "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            datacds.append(record)
    with open(lncrnafile, "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            datalncrna.append(record)
    
    min_size = min(len(datacds),len(datalncrna))
    print(min_size)
    print(type(datacds))
    #随机向下采样
    badatacds = balancedata(datacds, min_size)
    badatalncrna = balancedata(datalncrna, min_size)
    print(type(badatacds))

    #输出fasta数据
    SeqIO.write(badatacds,outputcdsfile,'fasta')
    SeqIO.write(badatalncrna,outputlncrnafile,'fasta')
    

def balancedata(data, size):
    if len(data) == size:
        remaining_data = np.array(data)
    else:
        idx = np.random.randint(len(data), size=size)
        remaining_data = np.array(data)[idx]
    
    return list(remaining_data)

#程序入口
if __name__ == '__main__':
    time_start=time.time()
    cdsfile = './data/Homo_sapiens.GRCh38.cds.hit.fa'
    lncrnafile = './data/Homo_sapiens.GRCh38.lncrna.hit.fa'
    outputcdsfile = './data/Homo_sapiens.GRCh38.cds.ba.fa'
    outputlncrnafile = './data/Homo_sapiens.GRCh38.lncrna.ba.fa'
    process(cdsfile, lncrnafile, outputcdsfile, outputlncrnafile)
    time_end=time.time()
    print('totally cost',time_end-time_start)


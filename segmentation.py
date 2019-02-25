#读取fasta数据，去除序列长度小于200的序列
import re
import sys
from Bio import SeqIO

def process(inputfile,outputfile):

    datatrain = []
    count = 0
    with open(inputfile, "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if len(record.seq) < 200:
                continue
            datatrain.append(record)

    SeqIO.write(datatrain,outputfile,'fasta')

# if len(sys.argv) !=3:
#     print("Use: filter_fasta.py <input> <output>");
#     exit(1)
# else:
#     process(sys.argv[1],sys.argv[2])
#程序入口
if __name__ == '__main__':
    inputfile = './data/Homo_sapiens.GRCh37.75.cds.all.fa'
    outputfile = './data/Homo_sapiens.GRCh37.75.cds.fa'
    process(inputfile= inputfile,outputfile = outputfile)
    print(1)
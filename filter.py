#GRCh38:cds计数104817 ncRNA计数37297，过滤后得到lncRNA

import re
import sys
from Bio import SeqIO

LONG_BIOTYPES = ['3prime_overlapping_ncrna', 'ambiguous_orf', 'antisense', 'antisense_rna', 'lincrna', 'ncrna_host',
                     'processed_transcript', 'sense_intronic', 'sense_overlapping']

def process(inputfile, outputfile):

    data = []
    regex = re.compile("\s+transcript_biotype:([a-zA-Z_0-9]+)")
    with open(inputfile, "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            r = regex.search(record.description)
            group = r.group(1).lower()
            if group in LONG_BIOTYPES:
                data.append(record)

    SeqIO.write(data,outputfile,'fasta')

# if len(sys.argv) !=3:
#     print("Use: filter_fasta.py <input> <output>");
#     exit(1)
# else:
#     process(sys.argv[1],sys.argv[2])
#程序入口
if __name__ == '__main__':
    inputfile = './data/Homo_sapiens.GRCh37.75.ncrna.fa'
    outputfile = './data/Homo_sapiens.GRCh37.75.lncrna.fa'
    process(inputfile,outputfile)
    print(1)


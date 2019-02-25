from Bio import SeqIO
import numpy as np
import re
import os


class SequenceAttributes:

    DI_TRI_PATTERNS = [
        "a",
        "c",
        "g",
        "t",
        "aa",
        "ac",
        "ag",
        "at",
        "ca",
        "cc",
        "cg",
        "ct",
        "ga",
        "gc",
        "gg",
        "gt",
        "ta",
        "tc",
        "tg",
        "tt",
        "aaa",
        "aac",
        "aag",
        "aat",
        "aca",
        "acc",
        "acg",
        "act",
        "aga",
        "agc",
        "agg",
        "agt",
        "ata",
        "atc",
        "atg",
        "att",
        "caa",
        "cac",
        "cag",
        "cat",
        "cca",
        "ccc",
        "ccg",
        "cct",
        "cga",
        "cgc",
        "cgg",
        "cgt",
        "cta",
        "ctc",
        "ctg",
        "ctt",
        "gaa",
        "gac",
        "gag",
        "gat",
        "gca",
        "gcc",
        "gcg",
        "gct",
        "gga",
        "ggc",
        "ggg",
        "ggt",
        "gta",
        "gtc",
        "gtg",
        "gtt",
        "taa",
        "tac",
        "tag",
        "tat",
        "tca",
        "tcc",
        "tcg",
        "tct",
        "tga",
        "tgc",
        "tgg",
        "tgt",
        "tta",
        "ttc",
        "ttg",
        "ttt"
    ]

    def __init__(self, input_file, clazz):
        self.fasta_file = input_file
        self.clazz = clazz

    def process(self, patterns=DI_TRI_PATTERNS):#提取特征
        data = []
        with open(self.fasta_file, "rU") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                data.append(self.attributes(record, self.clazz, patterns))

        dt = np.dtype(
            [("id", np.str_, 32), ("class", np.int_)] + [(pattern, np.float64) for pattern in
                                                                              patterns])#总特征
        self.data = np.asarray(np.array(data, dtype=dt))

        return self.data

    def attributes(self, record, type, patterns):
        seq = str(record.seq)
        attributes = [record.id, type]

        # 利用循环，计算每个特征值，添加到特征
        for pattern in patterns:
                attributes.append(self.count_pattern(pattern, seq))

        return tuple(attributes)

    #频率特征统计
    def count_pattern(self, pattern, seq):
        length = len(seq)
        count = len([m.start() for m in re.finditer("(?=%s)" % pattern, seq, re.IGNORECASE)])
        total = 0
        attr_length = len(pattern)
        for j in range(0, len(pattern)):
            total += int((length - j) / attr_length)

        return float(count) / float(total)

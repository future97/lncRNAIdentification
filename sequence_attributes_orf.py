from Bio import SeqIO
import numpy as np
import re
import os
import collections
import math
from random import randint

class SequenceAttributes:
    ALL_PATTERNS = [
                       "fl",  # ORF
                       "fp",
                       "ll",
                       "lp",
                       "a",  # 频率
                       "c",
                       "g",
                       "t",
                       "mi1",  # mi
                       "mi2",
                       "mi3",
                       "mi4",
                       "mi5",
                       "mi6",
                       "mi7",
                       "mi8",
                       "mi9",
                       "mi10",
                       "mi11",
                       "mi12",
                       "mi13",
                       "mi14",
                       "mi15",
                       "mi16",
                       "mi17",  # cmi
                       "cmi1",
                       "cmi2",
                       "cmi3",
                       "cmi4",
                       "cmi5",
                       "cmi6",
                       "cmi7",
                       "cmi8",
                       "cmi9",
                       "cmi10",
                       "cmi11",
                       "cmi12",
                       "cmi13",
                       "cmi14",
                       "cmi15",
                       "cmi16",
                       "cmi17",
                       "cmi18",
                       "cmi19",
                       "cmi20",
                       "cmi21",
                       "cmi22",
                       "cmi23",
                       "cmi24",
                       "cmi25",
                       "cmi26",
                       "cmi27",
                       "cmi28",
                       "cmi29",
                       "cmi30",
                       "cmi31",
                       "cmi32",
                       "cmi33",
                       "cmi34",
                       "cmi35",
                       "cmi36",
                       "cmi37",
                       "cmi38",
                       "cmi39",
                       "cmi40",
                       "cmi41",
                       "cmi42",
                       "cmi43",
                       "cmi44",
                       "cmi45",
                       "cmi46",
                       "cmi47",
                       "cmi48",
                       "cmi49",
                       "cmi50",
                       "cmi51",
                       "cmi52",
                       "cmi53",
                       "cmi54",
                       "cmi55",
                       "cmi56",
                       "cmi57",
                       "cmi58",
                       "cmi59",
                       "cmi60",
                       "cmi61",
                       "cmi62",
                       "cmi63",
                       "cmi64",
                       "cmi65",
                       "entropy2",  # entropy
                       "entropy3",
                       "kld1",  # KL散度
                       "kld2",
                       "kld3",
                       "markov",  # 马尔科夫特征
                       "getoentropy3",  # 广义拓扑熵 Generalized topological entropy
                       "getoentropy4",
                       "getoentropy5",
                       "toentropy3",  # 拓扑熵 Topological entropy
                       "toentropy4",
                       "toentropy5"
                   ],

    NOCMI_PATTERNS = [
        "fl",#ORF
        "fp",
        "ll",
        "lp",
        "mi1", #mi
        "mi2",
        "mi3",
        "mi4",
        "mi5",
        "mi6",
        "mi7",
        "mi8",
        "mi9",
        "mi10",
        "mi11",
        "mi12",
        "mi13",
        "mi14",
        "mi15",
        "mi16",
        "mi17",
        "entropy2", #entropy
        "entropy3",
        "kld1", #KL散度
        "kld2",
        "kld3",
        "getoentropy3", #广义拓扑熵 Generalized topological entropy
        "getoentropy4",
        "getoentropy5",
        "toentropy3", #拓扑熵 Topological entropy
        "toentropy4",
        "toentropy5"
    ]

    def __init__(self, input_file, clazz):
        self.fasta_file = input_file
        self.clazz = clazz

    def process(self, patterns=NOCMI_PATTERNS):#提取特征

        data = []
        with open(self.fasta_file, "rU") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                data.append(self.attributes(record, self.clazz, patterns))

        dt = np.dtype(
            [("id", np.str_, 32), ("class", np.int_), ("length", np.int_)] + [(pattern, np.float64) for pattern in
                                                                              patterns])#总特征
        self.data = np.asarray(np.array(data, dtype=dt))
        print("success")
        return self.data

    def attributes(self, record, type, patterns):
        first_orf_size = None
        longest_orf_size = None
        seq = str(record.seq)
        attributes = [record.id, type, len(seq)]

        # 利用循环，计算每个特征值，添加到特征
        for pattern in patterns:
            if pattern == "ll":
                if longest_orf_size == None:
                    longest_orf_size = self.longest_orf(seq)
                attributes.append(longest_orf_size)
            elif pattern == "lp":
                if longest_orf_size == None:
                    longest_orf_size = self.longest_orf(seq)
                attributes.append(float(longest_orf_size) / float(len(seq)))
            elif pattern == "fl":
                if first_orf_size == None:
                    first_orf_size = self.first_orf(seq)
                attributes.append(first_orf_size)
            elif pattern == "fp":
                if first_orf_size == None:
                    first_orf_size = self.first_orf(seq)
                attributes.append(float(first_orf_size) / float(len(seq)))

        #计算单个A,G,C,T频率 4维特征
        if 'a' in patterns: 
            frequencyValues, nucleotide_dict = self.count_frequency(seq)
            attributes.extend(frequencyValues)

        #计算互信息(Mutual Information) 17维特征
        if 'mi1' in patterns:
            miValues = self.computeMI(seq)
            attributes.extend(miValues)

        #计算条件互信息Conditional mutual information 65维特征
        if 'cmi1' in patterns:
            cmiValues = self.cmi(seq)
            attributes.extend(cmiValues)

        #计算2-mer和3-mer香农熵 2维
        if 'entropy2' in patterns:
            shannon_entropy = []
            shannon_entropy.append(self.shannon_sta(seq, 2))
            shannon_entropy.append(self.shannon_sta(seq, 3))
            attributes.extend(shannon_entropy)

        #计算相对熵 3维
        if 'kld1' in patterns:
            kldValues = self.computeKLValue(seq)
            attributes.extend(kldValues)

        #计算markov特征 1维
        if 'markov' in patterns:
            markovValues = self.getMarkovScore(seq, m=2)
            attributes.extend(markovValues)

        #计算广义拓扑熵 k=3,4,5 3d Generalized topological entropy
        if "getoentropy3" in patterns:
            getoentropyValues = []
            getoentropy3 = self.getoentropy(seq, 3)
            getoentropyValues.extend(getoentropy3)
            getoentropy4 = self.getoentropy(seq, 4)
            getoentropyValues.extend(getoentropy4)
            getoentropy5 = self.getoentropy(seq, 5)
            getoentropyValues.extend(getoentropy5)
            attributes.extend(getoentropyValues)
        
        #计算拓扑熵 k=3,4,5 3d topological entropy
        if "toentropy3" in patterns:
            toentropyValues = []
            toentropy3 = self.toentropy(seq, 3)
            toentropyValues.extend(toentropy3)
            toentropy4 = self.toentropy(seq, 4)
            toentropyValues.extend(toentropy4)
            toentropy5 = self.toentropy(seq, 5)
            toentropyValues.extend(toentropy5)
            attributes.extend(toentropyValues)

        return tuple(attributes)


    
    #功能函数
    #第一条ORF的长度,atg起始密码子，数据库获得的序列cds和lncRNA，是转换为AGCT的序列
    def first_orf(self, seq):
        index = re.search("atg", seq, re.IGNORECASE)
        if index == None:
            return 0
        else:
            return self.orf_size(index.start(), seq)
    
    #最长ORF的长度，atg
    def longest_orf(self, seq):
        sizes = []
        for index in [m.start() for m in re.finditer("atg", seq, re.IGNORECASE)]:
            sizes.append(self.orf_size(index, seq))
        if (len(sizes) == 0):
            return 0
        else:
            return max(sizes)
    #orf长度计算
    def orf_size(self, start, seq):
        for end in [m.start() for m in re.finditer("taa|tga|tag", seq, re.IGNORECASE)]:
            if end < start:
                continue

            length = end - start + 3
            if length % 3 == 0:
                return length

        length = len(seq[start:])
        length -= len(seq[start:]) % 3
        return length
    
    #函数功能：计算单个A,G,C,T频率
    def count_frequency(self,seq):
        AAList = ["A", "C", "G", "T"]
        nucleotide_dict = {'A': 0, 'C': 0, 'G': 0, 'T': 0}

        for i in range(len(seq)):
            if seq[i] in AAList:
                nucleotide_dict[seq[i]] += 1
            else:
                continue

        for item in nucleotide_dict.keys():
            nucleotide_dict[item] /= float(len(seq))
        sum = 0.0

        for item in nucleotide_dict.keys():
            sum += nucleotide_dict[item]

        result = []
        for AA in AAList:
            result.append(nucleotide_dict[AA])

        return result, nucleotide_dict
    
    #计算互信息(Mutual Information)
    def pairRepeateNumber(self,seq):
        di_nucleotides = {}
        for i in range(len(seq) - 1):
            di_seq = seq[i:i + 2]
            if di_seq not in di_nucleotides.keys():
                di_nucleotides[di_seq] = 1
            else:
                di_nucleotides[di_seq] += 1
        for item in di_nucleotides.keys():
            di_nucleotides[item] /= float(len(seq)-1)
        return di_nucleotides

    def mi(self,nucleotides, di_nucleotides, delta=0.001):
        # type: (object, object) -> object
        """
        :type nucleotides: dict
        :type di_nucleotides: dict
        """
        keys = nucleotides.keys()
        seqs = []
        mi = collections.OrderedDict()
        for item1 in keys:
            for item2 in keys:
                seq = item1 + item2
                seqs.append(seq)
                if seq in di_nucleotides.keys():
                    mi[seq] = di_nucleotides[seq] * math.log(((di_nucleotides[seq]+delta)/((nucleotides[item1]*nucleotides[item2])+delta)))
                else:
                    mi[seq] = 0.0
        value = 0.0
        for item in mi.keys():
            value += mi[item]
        mi["di_nucleotides"] = value

        di_nucleotideslist = []
        for item in mi.keys():
            di_nucleotideslist.append(mi.get(item))
        return di_nucleotideslist

    def computeMI(self,seq):
        di_nus = self.pairRepeateNumber(seq)
        frequencyValues, nucleotide_dict = self.count_frequency(seq)
        result = self.mi(nucleotide_dict, di_nus)
        return result
    
    #计算条件互信息
    def createDouble(self,List1):
        List2 = []
        for i in range(len(List1)):
            for j in range(len(List1)):
                item = List1[i] + List1[j]
                List2.append(item)
        return List2

    def createTrip(self,List1):
        trip = []
        for i in range(len(List1)):
            for j in range(len(List1)):
                for k in range(len(List1)):
                    item = List1[i] + List1[j] + List1[k]
                    trip.append(item)
        return trip

    def computeProb(self ,seq, subSeqList, delta=0.):
        prob = {}
        length = len(subSeqList[0])

        for item in subSeqList:
            count = seq.count(item)
            prob[item] = float("%.4f" % ((count + delta) / (len(seq) - length + 1 + 4 * delta)))
        return prob

    def compute_cmi(self,tri_prob, di_nucleotides, nucleotides, delta=0.001):
        tri_items = tri_prob.keys()
        cmi_list = []
        for tri_item in tri_items:
            if(tri_prob[tri_item] == 0.0):
                cmi_list.append(0.0)
            else:
                z = tri_item[2]
                xz = tri_item[0]+z
                yz = tri_item[1:]
                p_tri = tri_prob[tri_item]
                cmi_value = p_tri * math.log((nucleotides[z]*p_tri+delta) / (di_nucleotides[xz]*di_nucleotides[yz]+delta),2)
                cmi_list.append(cmi_value)
        cmi_list.append(np.sum(cmi_list))
        return cmi_list

    def cmi(self,seq):
        List1 = ["A", "T", "G", "C"]
        List2 = self.createDouble(List1)
        List3 = self.createTrip(List1)

        prob1 = self.computeProb(seq, List1)
        prob2 = self.computeProb(seq, List2)
        prob3 = self.computeProb(seq, List3)
        
        cmi_values = self.compute_cmi(prob3, prob2, prob1)
        
        return cmi_values

    def ctrateList(self,List1, n):
        if n == 2:
            return self.createDouble(List1)
        elif n == 3:
            return  self.createTrip(List1)
    
    #计算香农熵
    def shannon_sta(self,seq, n):
        List1 = ["A", "C", "G", "T"]
        listn = self.ctrateList(List1, n)
        probn = self.computeProb(seq, listn)
        sumpi = 0.
        for item in probn.keys():
            if probn[item] > 0:
                pi = probn[item] * math.log(probn[item], 2)
                sumpi += pi
        return -sumpi

    #计算相对熵，又称KL散度(Kullback–Leibler divergence)
    def computeKLD(self,probA={}, probB={}, delta = 0.001):
        score = 0.
        #不转换为list，就会报这个错误，解决办法就是先转换成list，再把需要的索引提取出来。
        list1 = list(probA.keys())
        list2 = list(probB.keys())
        for i in range(len(list1)):
            for j in range(len(list2)):
                px = probA[list1[i]]
                qx = probB[list2[j]]
                score += px * math.log((px + delta) / (qx + 4*delta), 2)
        return score

    def computeKLValue(self,seq):
        List1 = ["A", "T", "G", "C"]
        List2 = self.createDouble(List1)
        List3 = self.createTrip(List1)

        prob1 = self.computeProb(seq, List1)
        prob2 = self.computeProb(seq, List2)
        prob3 = self.computeProb(seq, List3)

        val1 = self.computeKLD(prob1, prob2)
        val2 = self.computeKLD(prob1, prob3)
        val3 = self.computeKLD(prob2, prob3)
        return  [val1, val2, val3]
    
    #计算马尔科夫Score
    def makeDict(self,text):
        #替换换行符和引号
        text = text.replace('\n', ' ')
        text = text.replace('\“', '')
        text = text.replace('\”', '')

        punc = ['，', '。', '？', '；', ':', '!']
        for symbol in punc:
            text = text.replace(symbol, ' '+symbol+' ')

        words = [word for word in text if word != '']

        wordict = {}
        for i in range(1, len(text)):
            if words[i-1] not in wordict:
                wordict[words[i-1]] = {}
            if words[i] not in wordict[words[i-1]]:
                wordict[words[i-1]][words[i]] = 0
            wordict[words[i-1]][words[i]] += 1

        return wordict

    def wordLen(self,wordict):
        sum = 0
        for key, value in wordict.items():
            sum += value
        return sum

    def retriveRandomWord(self,wordict):
        """
        计算每个单词的机率
        :param wordict:
        :return:
        """
        randindex = randint(1, self.wordLen(wordict))
        for key, value in wordict.items():
            randindex -= value
            if randindex <= 0:
                return key
    
    def generateNextChain(self,chain, outLen=200):
        wordict = self.makeDict(chain)
        currentword = chain[-1]
        
        for i in range(0, outLen):
            chain += currentword
            currentword = self.retriveRandomWord(wordict[currentword])
        return chain 

    def MarkovComputeProb(self,chain, subSeq, delta = 0.001):
        preSubSeq = subSeq[0:-1]
        countPreSubSeq = chain.count(preSubSeq)
        countSubSeq = chain.count(subSeq)
        prob = (countSubSeq + delta) / (countPreSubSeq + 4 * delta)
        return float("%.4f" % prob)

    def getMarkovScore(self,seq, m = 1, netxtLength=100, delta = 0.001):
        chain = self.generateNextChain(seq, netxtLength)
        L = len(chain)
        score = 0.
        for i in range(0, (L - m + 1)):
            subSeq = chain[i:i+m]
            subScore = self.MarkovComputeProb(chain, subSeq)
            probUp = self.MarkovComputeProb(chain, subSeq[0:-1]) * subScore
            probDown = (chain.count(subSeq[-1]) + delta)/ (L + 4 * delta)
            score += subScore * math.log(probUp / probDown, 2)
        
        return [score]

    #计算广义拓扑熵 Generalized topological entropy
    def getoentropy(self,w, k):
        #w是序列数据，或者是语句

        w = w.strip(' ')
        w_len = len(w)
        nw = 0
        nw = int(math.floor(math.log(w_len, 4)))
        sub_w = w[0:(4 ** nw) + nw]
        sub_w_len = len(sub_w)
        if ((k < 0 or k > nw)):
            return [1.0]

        ssum = 0.0000001
        for i in range(nw - k + 1, nw + 1):
            subset = set()
            pwi = 0
            ai = 0.0
            for j in range(0, sub_w_len - i + 1):
                subset.add(sub_w[j:j + i])
            pwi = len(subset)
            ai = ((math.log(pwi, 4)) / i)
            ssum += ai
        gtecvalue = ssum / k
        gtecvalues = []
        gtecvalues.append(gtecvalue)
        return gtecvalues
    
    #计算拓扑熵 Topological entropy
    def toentropy(self,w,k):
        w_len = len(w)
        nw = int(math.floor(math.log(w_len - 1, 4)))
        subw = w[0:(4 ** nw) + nw - 1]
        if k > nw:
            return [1.0]
        subwlen = len(subw)
        lk = []
        tpe = 0
        gtecvalue = 0
        for i in range(0, subwlen - k):
            if subw[i:i + k] not in lk:
                lk.append(subw[i:i + k])
        if len(lk) > 1:
            tpe = math.log(len(lk) - 1, 4)
        if k > 0:
            gtecvalue = tpe / k
        return [gtecvalue]



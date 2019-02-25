import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
#画出roc曲线图
def plotroc(oxgb_file,kmer_file,roc_file,title):
    oxgb_data = pd.read_csv(oxgb_file)
    kmer_data = pd.read_csv(kmer_file)
    oxgb_label = "Entropy(AUC = %.2f%%)" % (oxgb_data['auc'][0] * 100)
    kmer_label = "Kmer(AUC = %.2f%%)" % (kmer_data['auc'][0] * 100)
    fig, ax = plt.subplots()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(oxgb_data['false_positive_rate'], oxgb_data['true_positive_rate'],
     label=oxgb_label,color='coral', linestyle='-')
    plt.hold(True)
    ax.plot(kmer_data['false_positive_rate'], kmer_data['true_positive_rate'],
     label=kmer_label,color='blue', linestyle='-')


    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    ax.legend(loc=4)
    plt.savefig(roc_file, format='png')

def plotpr(oxgb_file,kmer_file,pr_file,title):
    oxgb_data = pd.read_csv(oxgb_file)
    kmer_data = pd.read_csv(kmer_file)
    oxgb_label = "Entropy(AUPR = %.2f%%)" % (oxgb_data['aupr'][0] * 100)
    kmer_label = "Kmer(AUPR = %.2f%%)" % (kmer_data['aupr'][0] * 100)
    fig, ax = plt.subplots()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(oxgb_data['recall'], oxgb_data['precision'],
     label=oxgb_label,color='coral', linestyle='-')
    plt.hold(True)
    ax.plot(kmer_data['recall'], kmer_data['precision'],
     label=kmer_label,color='blue', linestyle='-')


    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc=4)
    plt.savefig(pr_file, format='png')


if __name__ == '__main__':
    time_start=time.time()
    #roc文件数据
    orf_roc_file= './material/xgb/Homo_sapiens_GRCh38_orf_training_xgb_roc.csv'
    kmer_roc_file= './material/xgb/Homo_sapiens_GRCh38_kmer_training_xgb_roc.csv'
    #pr文件数据
    orf_pr_file= './material/xgb/Homo_sapiens_GRCh38_orf_training_xgb_pr.csv'
    kmer_pr_file= './material/xgb/Homo_sapiens_GRCh38_kmer_training_xgb_pr.csv'
    #保存文件路径
    roc_file = './material/xgb/Homo_sapiens_GRCh38_xgb_roc.png'
    pr_file = './material/xgb/Homo_sapiens_GRCh38_xgb_pr.png'
    #图片名称
    title_roc = 'Homo_sapiens_GRCh38_xgb_roc'
    title_pr = 'Homo_sapiens_GRCh38_xgb_pr'
    #画图函数
    plotroc(orf_roc_file, kmer_roc_file,roc_file,title_roc)
    plotpr(orf_pr_file, kmer_pr_file,pr_file,title_pr)
    time_end=time.time()
    print('totally cost',time_end-time_start)

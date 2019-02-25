import pandas as pd
import matplotlib.pyplot as plt
import time

def plot_feature_importance(feature_importance_file):
    base_name = feature_importance_file.split('.')[1].split('/')[2]
    df = pd.read_csv(feature_importance_file)
    df.plot(kind='barh', x = 'feature',y = 'fscore', legend=False,figsize=(12,15))
    plt.title('Feature Importance of GRCh38 Entropy', fontsize=25)
    plt.xlabel('relative importance', fontsize=25)
    plt.ylabel('feature', fontsize=25)
    figure_name = './material/%s.png' % base_name
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    plt.savefig(figure_name, format='png')

if __name__ == '__main__':
    time_start=time.time()
    feature_importance_file= './material/Homo_sapiens_GRCh38_orf_training_feature_importance.csv'
    plot_feature_importance(feature_importance_file)
    time_end=time.time()
    print('totally cost',time_end-time_start)
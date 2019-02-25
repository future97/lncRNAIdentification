import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from pathlib import Path
from xgb_attributes import XGBAttributes
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


def build_model(trainingfile, testingfile,log2c = ["9,11,2"],log2g = ["-3,-5,-2"]):
    datatrain = pd.read_csv(trainingfile)
    datatest = pd.read_csv(testingfile)
    base_name_select = trainingfile.split('.')[1].split('/')[2]
    base_name = trainingfile.split('.')[1].split('/')[2] + "_svm"#文件名称
    print("start train")
    feature_file = './material/%s.txt' % base_name_select
    feature_file_path = Path(feature_file)
    f = None
    if feature_file_path.is_file():
        file=open(feature_file, 'r')
        contents = file.readlines()
        for c in contents:
            f = c.split(' ')
            break
    else:
        xgbselect = XGBAttributes(datatrain,base_name_select)
        f = xgbselect.attributes()
        #保存选择的特征
        file = open(feature_file, 'w') 
        s = ' '.join(f) 
        file.write(s)
        file.close()

    print("features are: %s" % f)

    attributes  = datatrain[f]
    labels = datatrain["class"]
    testing_attributes = datatest[f]
    testing_labels = datatest["class"]
    model_file =  "./model/%s.plk" % base_name
    roc_csv_file = "./material/svm/%s_roc.csv" % base_name
    pr_csv_file = "./material/svm/%s_pr.csv" % base_name
    c, gamma =lnc_gridsearch(attributes,labels,log2c,log2g)
    
    # print("C=%.13f, Gamma=%.13f" % (c,gamma))
    # print("Building the model ...")
    clf = create_classifier(c=c, gamma=gamma)
    clf.fit(attributes, labels)
    joblib.dump(clf, model_file)
    # pre_testing_labels = clf.predict(testing_attributes)
    probabilities = clf.predict_proba(testing_attributes)
    long_probabilities = probabilities[:, 1]

    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(testing_labels,long_probabilities)
    precision, recall, _ = metrics.precision_recall_curve(testing_labels,long_probabilities)
    #x轴参数在前，y轴参数在后
    auc = metrics.auc(false_positive_rate, true_positive_rate)
    aupr = metrics.auc(recall, precision)
    accuracy, sensitivity, specificity, pre, reca, f1, mcc = accuracy_sensitivity_specificity(testing_labels,
                                                                                              long_probabilities)
    roc_data = pd.DataFrame()
    pr_data = pd.DataFrame()
    pr_data['precision'] = precision
    pr_data['recall'] = recall
    pr_data['aupr'] = None
    pr_data['aupr'][0] = aupr
    roc_data['false_positive_rate'] = false_positive_rate
    roc_data['true_positive_rate'] = true_positive_rate

    roc_data['auc'] = None
    roc_data['auc'][0] = auc

    roc_data['accuracy'] = None
    roc_data['accuracy'][0] = accuracy

    roc_data['sensitivity'] = None
    roc_data['sensitivity'][0] = sensitivity
    roc_data['specificity'] = None
    roc_data['specificity'][0] = specificity
    roc_data['precision'] = None
    roc_data['precision'][0] = pre
    roc_data['recall'] = None
    roc_data['recall'][0] = reca
    roc_data['f1'] = None
    roc_data['f1'][0] = f1
    roc_data['mcc'] = None
    roc_data['mcc'][0] = mcc
    roc_data.to_csv(roc_csv_file)
    pr_data.to_csv(pr_csv_file)


def lnc_gridsearch(X_train, y_train, log2c, log2g):
    print("Starting the SVM parameter search ...")
    c_begin, c_end, c_step = map(int, log2c[0].split(','))
    g_begin, g_end, g_step = map(int, log2g[0].split(','))
    c_final = []
    gamma_final = []
    for log2c in range(c_begin, c_end, c_step):
        for log2g in range(g_begin, g_end, g_step):
            c, gamma = 2 ** log2c, 2 ** log2g
            c_final.append(c)
            gamma_final.append(gamma)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_final, 'C': c_final}]
    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5)

    clf.fit(X_train, y_train)
    cv_result = pd.DataFrame.from_dict(clf.cv_results_)
    with open('./material/svmcv_result.csv', 'w') as f:
        cv_result.to_csv(f)
    print(clf.best_params_)
    return clf.best_params_['C'], clf.best_params_['gamma']
    # return clf.best_estimator_


# 创建分类器
def create_classifier(c, gamma, verbose=False, shrinking=True, probability=True):
    return svm.SVC(kernel='rbf', C=c, gamma=gamma, decision_function_shape='ovr', max_iter=-1, tol=0.001,
                   verbose=verbose, shrinking=shrinking, probability=probability)


# accuracy, sensitivity, specificity
def accuracy_sensitivity_specificity(labels, probabilities):
    pred = np.copy(probabilities)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    accuracy = metrics.accuracy_score(labels, pred)
    confusion_matrix = metrics.confusion_matrix(labels, pred)
    tp = confusion_matrix[1, 1]
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]
    sensitivity = float(tp) / (float(fn + tp) if float(fn + tp) > 0 else -1)
    specificity = float(tn) / (float(tn + fp) if float(tn + fp) > 0 else -1)
    precision = float(tp) / float((tp + fp) if float(tp + fp) > 0 else -1)
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall)
    MCC_up = float(tp * tn - fp * fn)
    MCC_down = float(np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = MCC_up / MCC_down
    return accuracy, sensitivity, specificity, precision, recall, f1, mcc

if __name__ == '__main__':
    time_start=time.time()
    trainingfile= './data/Homo_sapiens_GRCh37_kmer_training.csv'
    testingfile= './data/Homo_sapiens_GRCh37_kmer_testing.csv'
    build_model(trainingfile, testingfile)
    time_end=time.time()
    print('totally cost',time_end-time_start)

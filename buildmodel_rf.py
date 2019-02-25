import time
import numpy as np
import pandas as pd
from sklearn import metrics
from pathlib import Path
from xgb_attributes import XGBAttributes
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
#使用全特征跑模型
import sequence_attributes_kmer as sk
import sequence_attributes_orf as so
def build_model(trainingfile, testingfile):
    datatrain = pd.read_csv(trainingfile)
    datatest = pd.read_csv(testingfile)
    base_name_select = trainingfile.split('.')[1].split('/')[2]
    base_name = trainingfile.split('.')[1].split('/')[2] + "_rf"  # 文件名称
    print("start train")
    # feature_file = './material/%s.txt' % base_name_select
    # feature_file_path = Path(feature_file)
    # f = None
    # if feature_file_path.is_file():
    #     file = open(feature_file, 'r')
    #     contents = file.readlines()
    #     for c in contents:
    #         f = c.split(' ')
    #         break
    # else:
    #     xgbselect = XGBAttributes(datatrain, base_name_select)
    #     f = xgbselect.attributes()
    #     # 保存选择的特征
    #     file = open(feature_file, 'w')
    #     s = ' '.join(f)
    #     file.write(s)
    #     file.close()

    f = so.SequenceAttributes.NOCMI_PATTERNS
    # f = sk.SequenceAttributes.DI_TRI_PATTERNS   #kmer特征的全特征
    f.append('length')
    print("features are: %s" % f)

    attributes = datatrain[f]
    labels = datatrain["class"]
    testing_attributes = datatest[f]
    testing_labels = datatest["class"]
    model_file = "./model/%s.plk" % base_name
    roc_csv_file = "./material/rf/%s_roc.csv" % base_name
    pr_csv_file = "./material/rf/%s_pr.csv" % base_name
    #网格搜索
    clf = select_parameter(attributes, labels,len(f))
    joblib.dump(clf, model_file)
    # pre_testing_labels = clf.predict(testing_attributes)
    probabilities = clf.predict_proba(testing_attributes)
    long_probabilities = probabilities[:, 1]

    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(testing_labels, long_probabilities)
    precision, recall, _ = metrics.precision_recall_curve(testing_labels, long_probabilities)
    # x轴参数在前，y轴参数在后
    auc = metrics.auc(false_positive_rate, true_positive_rate)
    aupr = metrics.auc(recall, precision)
    accuracy, sensitivity, specificity, pre, reca, f1, mcc = accuracy_sensitivity_specificity(testing_labels,long_probabilities)
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

#效率低，不建议用
def rf_gridsearch(X_train, y_train,feature_length):
    print(feature_length)
    parameter_space = {
        "n_estimators": range(10,41,10),
        'max_depth': range(3, 11, 2),
        'min_samples_split': range(50, 100, 20),
        'min_samples_leaf': range(10, 40, 10),
        'max_features': range(1, feature_length, 2),
        "criterion": ["gini", "entropy"],
    }
    rf_model = RandomForestClassifier()
    clf = GridSearchCV(rf_model, parameter_space, cv=5)
    clf.fit(X_train, y_train)

    return clf.best_estimator_

def select_parameter(train_data, label,feature_length):
    print("Starting the randomforest parameter search ...")
    bint_model = RandomForestClassifier(min_samples_split=50,min_samples_leaf=20,max_depth=8,max_features='sqrt')
    param_test1 = {'n_estimators': range(10, 71, 10)}
    b1_model = grid_search_para(bint_model,param_test1,train_data,label,is_n_estimators = True)

    param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(50, 201, 20)}
    b2_model = grid_search_para(b1_model, param_test2, train_data, label)

    param_test3 = {'min_samples_split': range(80, 150, 20), 'min_samples_leaf': range(10, 60, 10)}
    b3_model = grid_search_para(b2_model, param_test3, train_data, label)

    param_test4 = {'max_features': range(1, feature_length, 2)}
    b4_model = grid_search_para(b3_model, param_test4, train_data, label)

    param_test5 = {"criterion": ["gini", "entropy"]}
    b5_model = grid_search_para(b4_model, param_test5, train_data, label)

    return b5_model

def grid_search_para(rf_model,para_grid,train_data,label,is_n_estimators = False):
    best_clf = None
    if not is_n_estimators:
        gridsearch = GridSearchCV(rf_model,para_grid,scoring='roc_auc',iid=False,cv=5)
        gridsearch.fit(train_data,label)
        print(gridsearch.estimator)
        best_clf = gridsearch.best_estimator_
    else:
        gridsearch = GridSearchCV(rf_model, para_grid, scoring='roc_auc', cv=5)
        gridsearch.fit(train_data, label)
        print(gridsearch.estimator)

        best_clf = gridsearch.best_estimator_


    return best_clf

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
    time_start = time.time()
    trainingfile = './data/Homo_sapiens_GRCh37_orf_training.csv'
    testingfile = './data/Homo_sapiens_GRCh37_orf_testing.csv'
    build_model(trainingfile, testingfile)
    time_end = time.time()
    print('totally cost', time_end - time_start)

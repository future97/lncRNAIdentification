import time
import pprint
import numpy as np
import pandas as pd
import xgboost as xgb 

from sklearn import metrics
from pathlib import Path
from xgb_attributes import XGBAttributes
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor#基于原生的xgboost接口，而XGBClassifier是基于sklearn的接口
from xgboost.sklearn import XGBClassifier
from sklearn.externals import joblib
#使用全特征跑模型
import sequence_attributes_kmer as sk
import sequence_attributes_orf as so
#xgboost参数选择
skfold = KFold(n_splits=5, shuffle=True)
 
def xgb_cv(clf, train_data, label, cv_folds=5, early_stopping_rounds=50, metric='rmse'):
    param = clf.get_xgb_params()
    train_data_cv = xgb.DMatrix(train_data, label)
    cv_res = xgb.cv(param, train_data_cv, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds, metrics=metric,
                    early_stopping_rounds=early_stopping_rounds)
    clf.set_params(n_estimators=cv_res.shape[0])

    clf.fit(train_data, label, eval_metric=metric)
 
    return cv_res.shape[0]
 
 
def grid_search_para(train_data, label, best_para=0, grid_param=0, is_search_estimator=False, search_lr=0.1,
                     scoring='roc_auc', search_estimators=100, iid=False, cv=skfold):
    if not is_search_estimator:
        print("search other parameters")
        xgb_ = XGBRegressor(**best_para)
        best_para['objective'] = 'binary:logistic'
        best_para['nthread'] = 8
        grid_search = GridSearchCV(estimator=xgb_, param_grid=grid_param, scoring=scoring, iid=iid, cv=cv)
        grid_search.fit(train_data, label)
        best_para.update(grid_search.best_params_)
    else:
        print("search n_estimators parameters")
        xgb_ = XGBRegressor(booster="dart")
        if best_para == 0:
            best_para = xgb_.get_params()
        best_para['n_estimators'] = search_estimators
        best_para['learning_rate'] = search_lr
        xgb_ = XGBRegressor(**best_para)
        best_estimator = xgb_cv(xgb_, train_data, label)
        best_para['n_estimators'] = best_estimator
 
    return best_para
 
 
def select_parameter(train_data, label):
    print("Starting the xgboost parameter search ...")
    best_para = grid_search_para(train_data, label, best_para=0,is_search_estimator=True)

    grid_param = {'max_depth': list(range(3, 10, 1)), 'min_child_weight': list(range(1, 12, 2))}
    best_para = grid_search_para(train_data, label, best_para, grid_param=grid_param)
 
    grid_param = {'gamma': [i / 10.0 for i in range(0, 5)]}
    best_para = grid_search_para(train_data, label, best_para, grid_param=grid_param)
 
    grid_param = {'subsample': [i / 10.0 for i in range(6, 10)], 'colsample_bytree': [i / 10.0 for i in range(6, 10)]}
    best_para = grid_search_para(train_data, label, best_para, grid_param=grid_param)
 
    grid_param = {'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 1e-5, 1e-2, 0.1, 1, 100]}
    best_para = grid_search_para(train_data, label, best_para, grid_param=grid_param)

    # best_para = grid_search_para(train_data, label, best_para, is_search_estimator=True, search_lr=0.1)
 
    return best_para

#全局调参，不可用
def xgb_GridSearch(X_train, y_train):
    grid_param = {'max_depth': list(range(3, 10, 1)),
                  'min_child_weight': list(range(1, 12, 2)),
                  'gamma': [i / 10.0 for i in range(0, 5)],
                  'subsample': [i / 10.0 for i in range(6, 10)],
                  'colsample_bytree': [i / 10.0 for i in range(6, 10)],
                  'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 1e-5, 1e-2, 0.1, 1, 100]
                  }
    xgb_model = xgb.XGBClassifier()
    clf = GridSearchCV(xgb_model, grid_param,cv=5)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    return clf.best_estimator_


#accuracy, sensitivity, specificity
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

def build_model(trainingfile, testingfile):
    datatrain = pd.read_csv(trainingfile)
    datatest = pd.read_csv(testingfile)
    base_name_select = trainingfile.split('.')[1].split('/')[2]
    base_name = trainingfile.split('.')[1].split('/')[2] + "_xgb"#文件名称
    print("start train")
    # feature_file = './material/%s.txt' % base_name_select
    # feature_file_path = Path(feature_file)
    # f = None
    # if feature_file_path.is_file():
    #     file=open(feature_file, 'r')
    #     contents = file.readlines()
    #     for c in contents:
    #         f = c.split(' ')
    #         break
    # else:
    #     xgbselect = XGBAttributes(datatrain,base_name_select)
    #     f = xgbselect.attributes()
    #     #保存选择的特征
    #     file = open(feature_file, 'w')
    #     s = ' '.join(f)
    #     file.write(s)
    #     file.close()

    # f = so.SequenceAttributes.NOCMI_PATTERNS
    f = sk.SequenceAttributes.DI_TRI_PATTERNS   #kmer特征的全特征
    # f.append('length')
    print("features are: %s" % f)

    attributes  = datatrain[f]
    labels = datatrain["class"]
    testing_attributes = datatest[f]
    testing_labels = datatest["class"]
    model_file =  "./model/%s.plk" % base_name
    roc_csv_file = "./material/xgb/%s_roc.csv" % base_name
    pr_csv_file = "./material/xgb/%s_pr.csv" % base_name
    #参数选择
    best_para=select_parameter(attributes,labels)
    pprint.pprint("The best parameter is \n {}".format(best_para))
    # 训练模型
    clf = XGBClassifier(
        max_depth = best_para['max_depth'],
        min_child_weight = best_para['min_child_weight'],
        gamma = best_para['gamma'],
        subsample = best_para['subsample'],
        colsample_bytree = best_para['colsample_bytree'],
        reg_alpha = best_para['reg_alpha'],
        objective = best_para['objective'],
        learning_rate = best_para['learning_rate']
    )
    clf.fit(attributes, labels)
    joblib.dump(clf, model_file)
    #
    probabilities = clf.predict_proba(testing_attributes)
    long_probabilities = probabilities[:, 1]

    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(testing_labels,long_probabilities)
    precision, recall, _ = metrics.precision_recall_curve(testing_labels,long_probabilities)
    #x轴参数在前，y轴参数在后
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


if __name__ == '__main__':
    time_start=time.time()
    trainingfile = './data/Homo_sapiens_GRCh37_kmer_training.csv'
    testingfile = './data/Homo_sapiens_GRCh37_kmer_testing.csv'
    build_model(trainingfile, testingfile)
    time_end=time.time()
    print('totally cost',time_end-time_start)

import pandas as pd
import xgboost as xgb 
import operator
import matplotlib.pyplot as plt

class XGBAttributes:
    def __init__(self, data,base_name):
        self.data = data
        self.base_name = base_name

    def create_feature_map(self,features,feature_map):
        
        outfile = open(feature_map,'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i,feat))
            i = i+1
        outfile.close()

    def attributes(self):
        
        params = {
            'booster':'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth':4,
            'lambda':10,
            'subsample':0.75,
            'colsample_bytree':0.75,
            'min_child_weight':2,
            'eta': 0.025,
            'seed':0,
            'nthread':8,
            'silent':1
        }

        round = 5
        y = self.data["class"]
        X = self.data.drop(["class","id"],1)
        feature_map = './material/%s_xgb.map' % self.base_name
        xgtrain = xgb.DMatrix(X,label=y)
        bst = xgb.train(params,xgtrain,num_boost_round=round)

        features = [x for x in self.data.columns if x not in ["id","class"]]
        self.create_feature_map(features,feature_map)

        importance = bst.get_fscore(fmap=feature_map)
        importance = sorted(importance.items(),key = operator.itemgetter(1))
        df = pd.DataFrame(importance,columns=['feature','fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        df.to_csv("./material/%s_feature_importance.csv" % self.base_name, index=False)

        # df.plot(kind='barh', x = 'feature',y = 'fscore', legend=False,figsize=(6,10))
        # plt.title('%s Feature Importance' % self.base_name)
        # plt.xlabel('relative importance')
        # figure_name = './material/%s_feature_weight.eps' % self.base_name
        # plt.savefig(figure_name, format='eps')
        return df['feature'].tolist()
 

    

import numpy as np
import pandas as pd
import scipy as sp
import lightgbm as lgb
import gc
import datetime
import random
import scipy.special as special
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

rawpath='C:\\final\\'
temppath='C:\\final\\temp\\'
iapath='C:\\final\\temp\\installedactions\\'

def logloss(act, preds):
    epsilon = 1e-15
    preds = sp.maximum(epsilon, preds)
    preds = sp.minimum(1 - epsilon, preds)
    ll = sum(act * sp.log(preds) + sp.subtract(1, act) * sp.log(sp.subtract(1, preds)))
    ll = ll * -1.0 / len(act)
    return ll


def getTrainVal(X_train, scope=(28, 29), val_type='30', seed=1000):
    if val_type == '30':
        X_val = X_train.loc[X_train['day'] == 30, :]
        X_train = X_train.loc[(X_train['day'] >= scope[0]) & (X_train['day'] <= scope[1]), :]
    elif val_type == '73':
        X_train = X_train.loc[(X_train['day'] >= scope[0]) & (X_train['day'] <= scope[1]), :]
        X_train, X_val, y_train, y_val = train_test_split(X_train, X_train['label'], test_size=0.3, random_state=seed)
    return X_train, X_val


t_start = datetime.datetime.now()
X_loc_train=pd.read_csv(temppath+'2_smooth.csv')
print('load train over...')
X_loc_test=pd.read_csv(temppath+'2_test_smooth.csv')
print('load test over...')

##########################################################CV预测时30号验证效果不好，而其中一折做验证来提前停止，后面删除30天数据
X_loc_train, X_loc_val = getTrainVal(X_loc_train, scope=(28, 29), val_type='30', seed=1000)

drop = ['label', 'day']
y_loc_train = X_loc_train.loc[:, 'label']
X_loc_train.drop(drop, axis=1, inplace=True)

# y_loc_val = X_loc_val.loc[:, 'label']
# X_loc_val.drop(drop, axis=1, inplace=True)

res = X_loc_test.loc[:, ['instanceID']]
X_loc_test.drop(['instanceID'], axis=1, inplace=True)
X_loc_test.drop(drop, axis=1, inplace=True)


gc.collect()
print('preprocess over...', X_loc_train.shape)

##########################################################比赛只用了lightGBM单模型
X_loc_train=X_loc_train.values
y_loc_train=y_loc_train.values
# X_loc_val=X_loc_val.values
# y_loc_val=y_loc_val.values
X_loc_test=X_loc_test.values

##########################################################交叉预测，实际上是stacking第一层做的操作
# 利用不同折数加参数，特征，样本（随机数种子）扰动，再加权平均得到最终成绩
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=29, max_depth=-1, learning_rate=0.1, n_estimators=10000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=1, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, nthread=-1, silent=True)
del X_loc_val

skf=list(StratifiedKFold(y_loc_train, n_folds=10, shuffle=True, random_state=1024))
for i, (train, test) in enumerate(skf):
    print("Fold", i)
    model.fit(X_loc_train[train], y_loc_train[train], eval_metric='logloss',eval_set=[(X_loc_train[train], y_loc_train[train]), (X_loc_train[test], y_loc_train[test])],early_stopping_rounds=100)
    preds= model.predict_proba(X_loc_test, num_iteration=model.best_iteration)[:, 1]
    print('mean:', preds.mean())
    res['prob_%s' % str(i)] = preds

#平均或者加权的方式有很多种，台大三傻的比赛分享里有一个利用sigmoid反函数来平均的方法效果不错
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
print(now)
res.sort_values("instanceID", ascending=True, inplace=True)
res.to_csv(rawpath+"%s.csv" % now, index=False)

t_end = datetime.datetime.now()
print('training time: %s' % ((t_end - t_start).seconds/60))

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import scipy as sp
import matplotlib.pyplot as plt
import gc
import datetime
import random
import scipy.special as special
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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


class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)


def readData(m_type='inner', scope=(28, 30)):  ###################left merge不改变顺序会比inner merge差1到2个万分点
    X_train = pd.read_csv(rawpath+'train.csv')
    pos=(X_train['clickTime']//10000000>=scope[0]).values & (X_train['clickTime']//10000000<=scope[1]).values
    X_train=X_train.loc[pos,:]
    X_test = pd.read_csv(rawpath+'test.csv')
    X_train.drop('conversionTime', axis=1, inplace=True)

    userfile = pd.read_csv(rawpath+'user.csv')
    X_train = X_train.merge(userfile, how=m_type, on='userID')
    X_test = X_test.merge(userfile, how=m_type, on='userID')
    del userfile
    gc.collect()

    adfile = pd.read_csv(rawpath+'ad.csv')
    X_train = X_train.merge(adfile, how=m_type, on='creativeID')
    X_test = X_test.merge(adfile, how=m_type, on='creativeID')
    del adfile
    gc.collect()

    appcatfile = pd.read_csv(rawpath+'app_categories.csv')
    X_train = X_train.merge(appcatfile, how=m_type, on='appID')
    X_test = X_test.merge(appcatfile, how=m_type, on='appID')
    del appcatfile
    gc.collect()

    positionfile = pd.read_csv(rawpath+'position.csv')
    X_train = X_train.merge(positionfile, how=m_type, on='positionID')
    X_test = X_test.merge(positionfile, how=m_type, on='positionID')
    del positionfile
    gc.collect()
    print('merge type:', m_type)
    return X_train, X_test

##################################重复数据Trick，初赛有3.5个千分点提升，决赛在原始数据的基础上有3个千分点提升
#训练集上的情况也会在测试集上出现
def doTrick(data):
    subset = ['creativeID', 'positionID', 'adID', 'appID', 'userID']
    data['maybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 3

    #比较关键的一步，初赛刚发现trick时提升不多,经过onehot后提升近3个千分点
    features_trans = ['maybe']
    data = pd.get_dummies(data, columns=features_trans)
    data['maybe_0'] = data['maybe_0'].astype(np.int8)
    data['maybe_1'] = data['maybe_1'].astype(np.int8)
    data['maybe_2'] = data['maybe_2'].astype(np.int8)
    data['maybe_3'] = data['maybe_3'].astype(np.int8)

    #时间差Trick
    temp = data.loc[:,['clickTime', 'creativeID', 'positionID', 'adID', 'appID', 'userID']].drop_duplicates(subset=subset, keep='first')
    # temp = temp.drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'clickTime': 'diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['diffTime_first'] = data['clickTime'] - data['diffTime_first']
    del temp,pos
    gc.collect()
    temp = data.loc[:,['clickTime', 'creativeID', 'positionID', 'adID', 'appID', 'userID']].drop_duplicates(subset=subset, keep='last')
    # temp = temp.drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'clickTime': 'diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['diffTime_last'] = data['diffTime_last'] - data['clickTime']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['diffTime_first', 'diffTime_last']] = -1 #置0会变差

    #重复次数是否大于2
    temp=data.groupby(subset)['label'].count().reset_index()
    temp.columns=['creativeID', 'positionID', 'adID', 'appID', 'userID','large2']
    temp['large2']=1*(temp['large2']>2)
    data = pd.merge(data, temp, how='left', on=subset)
    #-----------
    # data['last_click'] = data['clickTime']
    # pos = data.duplicated(subset=subset, keep=False)
    # data.loc[pos, 'last_click'] = data.loc[pos, 'last_click'].diff(periods=1)
    # pos = ~data.duplicated(subset=subset, keep='first')
    # data.loc[pos, 'last_click'] = -1
    # data['next_click'] = data['clickTime']
    # pos = data.duplicated(subset=subset, keep=False)
    # data.loc[pos, 'next_click'] = -1 * data.loc[pos, 'next_click'].diff(periods=-1)
    # pos = ~data.duplicated(subset=subset, keep='last')
    # data.loc[pos, 'next_click'] = -1
    # del pos
    # data['maybe_4']=data['maybe_1']+data['maybe_2']
    # data['maybe_5']=data['maybe_1']+data['maybe_3']
    # data['diffTime_span']=data['diffTime_last']+data['diffTime_first']
    #-------------
    del temp
    gc.collect()
    return data

##################################Trick2基于userID重复的数据做，重要性高但是线上效果不好，和Trick信息重复了
def doTrick2(X_train,X_test):
    res = X_test[['instanceID']]
    X_test.drop('instanceID', axis=1, inplace=True)
    data = X_train.append(X_test, ignore_index=True)
    del X_train, X_test
    gc.collect()

    subset = ['userID']
    data['umaybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'umaybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'umaybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'umaybe'] = 3
    del pos
    gc.collect()
    features_trans = ['umaybe']
    data = pd.get_dummies(data, columns=features_trans)
    data['umaybe_0'] = data['umaybe_0'].astype(np.int8)
    data['umaybe_1'] = data['umaybe_1'].astype(np.int8)
    data['umaybe_2'] = data['umaybe_2'].astype(np.int8)
    data['umaybe_3'] = data['umaybe_3'].astype(np.int8)

    temp = data[['clickTime','userID']]
    temp = temp.drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'clickTime': 'udiffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['udiffTime_first'] = data['clickTime'] - data['udiffTime_first']
    del temp
    gc.collect()
    temp = data[['clickTime','userID']]
    temp = temp.drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'clickTime': 'udiffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['udiffTime_last'] = data['udiffTime_last'] - data['clickTime']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['udiffTime_first', 'udiffTime_last']] = -1

    X_train = data.loc[data['label'] != -1, :]
    X_test = data.loc[data['label'] == -1, :]
    X_test.loc[:, 'instanceID'] = res.values
    del temp,data
    gc.collect()
    return X_train, X_test


def doPre(data):
    data['day'] = data['clickTime'] // 1000000
    data['hour'] = data['clickTime'] % 1000000 // 10000
    # data['clickTime'] = data['day'] * 1440 + (data['clickTime'] % 1000000 // 10000) * 60 + (data['clickTime'] % 10000 // 100) * 60 + data['clickTime'] % 100  # 默认
    # data['clickTime'] = data['day'] * 1440 + (data['clickTime'] % 1000000 // 10000) * 60 + data['clickTime'] % 10000#best

    # data['week'] = data['day'] % 7

    # data['appCategory_main'] = data['appCategory']
    # data.loc[data['appCategory'] > 99, 'appCategory_main'] = data.loc[data['appCategory'] > 99, 'appCategory'] // 100
    # data['appCategory'] = data['appCategory'] % 100

    # data.loc[data['age'] < 10,'age']=0
    # data.loc[(data['age'] >= 10)&(data['age']< 18), 'age'] = 1
    # data.loc[(data['age'] >= 18) & (data['age'] < 24), 'age'] = 2
    # data.loc[(data['age'] >= 24) & (data['age'] < 30), 'age'] = 3
    # data.loc[(data['age'] >= 30) & (data['age'] < 40), 'age'] = 4
    # data.loc[(data['age'] >= 40) & (data['age'] < 60), 'age'] = 5
    # data.loc[data['age'] >= 60, 'age'] = 6

    # data.loc[(data['hour'] >= 8) & (data['hour'] <14 ), 'preiod'] = 0
    # data.loc[(data['hour'] >= 14) | (data['hour'] < 8), 'preiod'] = 1
    # data = pd.get_dummies(data, columns=['preiod'])
    return data

##################################均值特征
def doAvg(X_train, X_test):
    res = X_test[['instanceID']]
    X_test.drop('instanceID', axis=1, inplace=True)
    data = X_train.append(X_test, ignore_index=True)
    del X_train, X_test
    gc.collect()

    # 小时均值特征
    grouped = data.groupby('userID')['hour'].mean().reset_index()
    grouped.columns = ['userID', 'user_mean_hour']
    data = data.merge(grouped, how='left', on='userID')
    grouped = data.groupby('appID')['hour'].mean().reset_index()
    grouped.columns = ['appID', 'app_mean_hour']
    data = data.merge(grouped, how='left', on='appID')
    grouped = data.groupby('appCategory')['hour'].mean().reset_index()
    grouped.columns = ['appCategory', 'appCategory_mean_hour']
    data = data.merge(grouped, how='left', on='appCategory')
    grouped = data.groupby('positionID')['hour'].mean().reset_index()
    grouped.columns = ['positionID', 'position_mean_hour']
    data = data.merge(grouped, how='left', on='positionID')

    # 年龄均值特征
    grouped = data.groupby('appID')['age'].mean().reset_index()
    grouped.columns = ['appID', 'app_mean_age']
    data = data.merge(grouped, how='left', on='appID')
    grouped = data.groupby('positionID')['age'].mean().reset_index()
    grouped.columns = ['positionID', 'position_mean_age']
    data = data.merge(grouped, how='left', on='positionID')
    grouped = data.groupby('appCategory')['age'].mean().reset_index()
    grouped.columns = ['appCategory', 'appCategory_mean_age']
    data = data.merge(grouped, how='left', on='appCategory')
    # grouped = data.groupby('creativeID')['age'].mean().reset_index()
    # grouped.columns = ['creativeID', 'creative_mean_age']
    # data = data.merge(grouped, how='left', on='creativeID')
    # grouped = data.groupby('adID')['age'].mean().reset_index()
    # grouped.columns = ['adID', 'ad_mean_age']
    # data = data.merge(grouped, how='left', on='adID')

    X_train = data.loc[data['label'] != -1, :]
    X_test = data.loc[data['label'] == -1, :]
    X_test.loc[:, 'instanceID'] = res.values
    del data, grouped
    gc.collect()
    return X_train, X_test

##################################活跃数特征
def doActive(X_train, X_test):
    res = X_test[['instanceID']]
    X_test.drop('instanceID', axis=1, inplace=True)
    data = X_train.append(X_test, ignore_index=True)
    del X_train, X_test
    gc.collect()

    # 活跃特征选取类别多的，类别太少，nunique差别不大,广告随时都在，用户不是时刻都在,一个只出现一次的用户活跃的ad,app,advertiser,camgaign,creative都为1
    # 用户活跃小时数
    add = pd.DataFrame(data.groupby(["userID"]).hour.nunique()).reset_index()
    add.columns = ["userID", "user_active_hour"]
    data = data.merge(add, on=["userID"], how="left")

    # 活跃app数特征
    add = pd.DataFrame(data.groupby(["appCategory"]).appID.nunique()).reset_index()
    add.columns = ["appCategory", "appCategory_active_app"]
    data = data.merge(add, on=["appCategory"], how="left")
    # add = pd.DataFrame(data.groupby(["userID"]).appID.nunique()).reset_index()
    # add.columns = ["userID", "user_active_app"]
    # data = data.merge(add, on=["userID"], how="left")
    # add = pd.DataFrame(data.groupby(["age"]).appID.nunique()).reset_index()
    # add.columns = ["age", "age_active_app"]
    # data = data.merge(add, on=["age"], how="left")
    # add = pd.DataFrame(data.groupby(["sitesetID"]).appID.nunique()).reset_index()
    # add.columns = ["sitesetID", "siteset_active_app"]
    # data = data.merge(add, on=["sitesetID"], how="left")
    # add = pd.DataFrame(data.groupby(["positionType"]).appID.nunique()).reset_index()
    # add.columns = ["positionType", "positionType_active_app"]
    # data = data.merge(add, on=["positionType"], how="left")
    # add = pd.DataFrame(data.groupby(["positionID"]).appID.nunique()).reset_index()
    # add.columns = ["positionID", "position_active_app"]
    # data = data.merge(add, on=["positionID"], how="left")
    add = pd.DataFrame(data.groupby(["connectionType"]).appID.nunique()).reset_index()
    add.columns = ["connectionType", "connectionType_active_app"]
    data = data.merge(add, on=["connectionType"], how="left")

    # 活跃position数特征
    add = pd.DataFrame(data.groupby(["appID"]).positionID.nunique()).reset_index()
    add.columns = ["appID", "app_active_position"]
    data = data.merge(add, on=["appID"], how="left")
    add = pd.DataFrame(data.groupby(["appCategory"]).positionID.nunique()).reset_index()
    add.columns = ["appCategory", "appCategory_active_position"]
    data = data.merge(add, on=["appCategory"], how="left")
    # add = pd.DataFrame(data.groupby(["userID"]).positionID.nunique()).reset_index()
    # add.columns = ["userID", "user_active_position"]
    # data = data.merge(add, on=["userID"], how="left")
    # add = pd.DataFrame(data.groupby(["age"]).positionID.nunique()).reset_index()
    # add.columns = ["age", "age_active_position"]
    # data = data.merge(add, on=["age"], how="left")
    # add = pd.DataFrame(data.groupby(["positionType"]).positionID.nunique()).reset_index()
    # add.columns = ["positionType", "positionType_active_position"]
    # data = data.merge(add, on=["positionType"], how="left")
    # add = pd.DataFrame(data.groupby(["advertiserID"]).positionID.nunique()).reset_index()
    # add.columns = ["advertiserID", "advertiser_active_position"]
    # data = data.merge(add, on=["advertiserID"], how="left")

    #活跃user数特征
    add = pd.DataFrame(data.groupby(["appID"]).userID.nunique()).reset_index()
    add.columns = ["appID", "app_active_user"]
    data = data.merge(add, on=["appID"], how="left")
    add = pd.DataFrame(data.groupby(["positionID"]).userID.nunique()).reset_index()
    add.columns = ["positionID", "position_active_user"]
    data = data.merge(add, on=["positionID"], how="left")
    add = pd.DataFrame(data.groupby(["appCategory"]).userID.nunique()).reset_index()
    add.columns = ["appCategory", "appCategory_active_user"]
    data = data.merge(add, on=["appCategory"], how="left")

    add = pd.DataFrame(data.groupby(["userID"]).creativeID.nunique()).reset_index()
    add.columns = ["userID", "user_active_creative"]
    data = data.merge(add, on=["userID"], how="left")
    # add = pd.DataFrame(data.groupby(["userID"]).sitesetID.nunique()).reset_index()
    # add.columns = ["userID", "user_active_siteset"]
    # data = data.merge(add, on=["userID"], how="left")
    # add = pd.DataFrame(data.groupby(["userID"]).appCategory.nunique()).reset_index()
    # add.columns = ["userID", "user_active_appCategory"]
    # data = data.merge(add, on=["userID"], how="left")
    add = pd.DataFrame(data.groupby(["positionID"]).advertiserID.nunique()).reset_index()
    add.columns = ["positionID", "positionID_active_advertiser"]
    data = data.merge(add, on=["positionID"], how="left")


    X_train = data.loc[data['label'] != -1, :]
    X_test = data.loc[data['label'] == -1, :]
    X_test.loc[:, 'instanceID'] = res.values
    del data, add
    gc.collect()
    return X_train, X_test

##################################这几个操作尝试过，效果不佳，后来放弃了
def doOneHot(X_train, X_test):
    res = X_test[['instanceID']]
    X_test.drop('instanceID', axis=1, inplace=True)
    data = X_train.append(X_test, ignore_index=True)
    del X_train, X_test
    gc.collect()

    features_trans = ['gender','appCategory_main','connectionType']
    data = pd.get_dummies(data, columns=features_trans)

    X_train = data.loc[data['label'] != -1, :]
    X_test = data.loc[data['label'] == -1, :]
    X_test.loc[:, 'instanceID'] = res.values
    del data
    gc.collect()
    return X_train, X_test
def doCrossProduct(data):
    data['position_creative'] = data['positionID'] * data['creativeID']
    data['creative_age'] = data['creativeID'] * data['age']
    return data
def doDescartes(X_train, X_test):
    res = X_test[['instanceID']]
    X_test.drop('instanceID', axis=1, inplace=True)
    data = X_train.append(X_test, ignore_index=True)
    del X_train, X_test
    gc.collect()

    for feat_1 in ['maybe_0', 'maybe_2']:
        for feat_2 in ['connectionType', 'creativeID', 'positionID']:
            le = LabelEncoder()
            data[feat_1 + '_' + feat_2] = le.fit_transform(data[feat_1].astype('str') + data[feat_2].astype('str'))
    X_train = data.loc[data['label'] != -1, :]
    X_test = data.loc[data['label'] == -1, :]
    X_test.loc[:, 'instanceID'] = res.values
    del data
    gc.collect()
    return X_train, X_test
def doSpecial(X_train, X_test):
    res = X_test[['instanceID']]
    X_test.drop('instanceID', axis=1, inplace=True)
    data = X_train.append(X_test, ignore_index=True)
    del X_train, X_test
    gc.collect()

    #####增加id与时间的斜率
    Min_id = data["listing_id"].min()
    Min_time = data["time"].min()
    data["gradient"] = ((data["listing_id"]) - Min_id) / (data["time"] - Min_time)

    X_train = data.loc[data['label'] != -1, :]
    X_test = data.loc[data['label'] == -1, :]
    X_test.loc[:, 'instanceID'] = res.values
    del data
    gc.collect()
    return X_train, X_test


X_loc_train,X_loc_test=readData(m_type='inner',scope=(28,30))
print('readData over')
X_loc_train=doPre(X_loc_train)
X_loc_test=doPre(X_loc_test)
print('doPre over...')

##########################################################actions和installed文件特征
temp = pd.read_csv(iapath+'all_app_seven_day_cnt.csv')
X_loc_train = pd.merge(X_loc_train, temp, how='left', on=['appID', 'day'])
X_loc_test = pd.merge(X_loc_test, temp, how='left', on=['appID', 'day'])
temp = pd.read_csv(iapath+'all_user_seven_day_cnt.csv')
X_loc_train = pd.merge(X_loc_train, temp, how='left', on=['userID', 'day'])
X_loc_test = pd.merge(X_loc_test, temp, how='left', on=['userID', 'day'])
temp = pd.read_csv(iapath+'userInstalledappscount.csv')
X_loc_train = pd.merge(X_loc_train, temp, how='left', on=['userID'])
X_loc_test = pd.merge(X_loc_test, temp, how='left', on=['userID'])
temp = pd.read_csv(iapath+'appInstalledusercount.csv')
X_loc_train = pd.merge(X_loc_train, temp, how='left', on=['appID'])
X_loc_test = pd.merge(X_loc_test, temp, how='left', on=['appID'])
temp = pd.read_csv(iapath+'ageuserInstalledappscount.csv')
X_loc_train = pd.merge(X_loc_train, temp, how='left', on=['age'])
X_loc_test = pd.merge(X_loc_test, temp, how='left', on=['age'])
temp = pd.read_csv(iapath+'appCatInstalledusercount.csv')
X_loc_train = pd.merge(X_loc_train, temp, how='left', on=['appCategory'])
X_loc_test = pd.merge(X_loc_test, temp, how='left', on=['appCategory'])
temp = pd.read_csv(iapath+'eduuserInstalledappscount.csv')
X_loc_train = pd.merge(X_loc_train, temp, how='left', on=['education'])
X_loc_test = pd.merge(X_loc_test, temp, how='left', on=['education'])
temp = pd.read_csv(iapath+'genderuserInstalledappscount.csv')
X_loc_train = pd.merge(X_loc_train, temp, how='left', on=['gender'])
X_loc_test = pd.merge(X_loc_test, temp, how='left', on=['gender'])

##########################################################appID平均回流时间特征
temp = pd.read_csv(temppath+'app_cov_diffTime.csv')
X_loc_train = pd.merge(X_loc_train, temp, how='left', on=['appID'])
X_loc_test = pd.merge(X_loc_test, temp, how='left', on=['appID'])
temp = pd.read_csv(temppath+'appCat_cov_diffTime.csv')
X_loc_train = pd.merge(X_loc_train, temp, how='left', on=['appCategory'])
X_loc_test = pd.merge(X_loc_test, temp, how='left', on=['appCategory'])
X_loc_train['cov_diffTime'].fillna(value=X_loc_train['appCat_cov_diffTime'], inplace=True)
X_loc_test['cov_diffTime'].fillna(value=X_loc_test['appCat_cov_diffTime'], inplace=True)
X_loc_train.drop(['appCat_cov_diffTime'],axis=1,inplace=True)
X_loc_test.drop(['appCat_cov_diffTime'],axis=1,inplace=True)
print('app_cov_diffTime over...')

##########################################################活跃数特征
X_loc_train,X_loc_test=doActive(X_loc_train,X_loc_test)
print('doActive over...')

##########################################################均值特征
X_loc_train,X_loc_test=doAvg(X_loc_train,X_loc_test)
print('doAvg over...')

print(X_loc_train.shape)
print(X_loc_train.columns)
# res = X_loc_test[['instanceID']]
# X_loc_test.drop('instanceID', axis=1, inplace=True)
# data = X_loc_train.append(X_loc_test, ignore_index=True)
# del X_loc_train, X_loc_test
# gc.collect()
# # data.sort_values(['userID','clickTime'],inplace=True,kind='mergesort')
# # data['ulast_click']=data['clickTime']
# # pos=data.duplicated(subset=['userID'], keep=False)
# # data.loc[pos,'ulast_click']=data.loc[pos,'ulast_click'].diff(periods=1)
# # pos=~data.duplicated(subset=['userID'], keep='first')
# # data.loc[pos,'ulast_click']=-1
# # data['unext_click']=data['clickTime']
# # pos=data.duplicated(subset=['userID'], keep=False)
# # data.loc[pos,'unext_click']=-1*data.loc[pos,'unext_click'].diff(periods=-1)
# # pos=~data.duplicated(subset=['userID'], keep='last')
# # data.loc[pos,'unext_click']=-1
# # del pos
# # temp = data.loc[:, ['clickTime',  'userID']].drop_duplicates(subset=['userID'],keep='first')
# # temp.rename(columns={'clickTime': 'udiffTime_first'}, inplace=True)
# # data = pd.merge(data, temp, how='left', on=['userID'])
# # data['udiffTime_first'] = data['clickTime'] - data['udiffTime_first']
# # del temp
# # gc.collect()
# # temp = data.loc[:, ['clickTime', 'userID']].drop_duplicates(subset=['userID'],keep='last')
# # temp.rename(columns={'clickTime': 'udiffTime_last'}, inplace=True)
# # data = pd.merge(data, temp, how='left', on=['userID'])
# # data['udiffTime_last'] = data['udiffTime_last'] - data['clickTime']
# # del temp
# # gc.collect()
# # data.loc[~data.duplicated(subset=['userID'], keep=False), ['udiffTime_first', 'udiffTime_last']] = -1
#
# X_loc_train = data.loc[data['label'] != -1, :]
# X_loc_test = data.loc[data['label'] == -1, :]
# X_loc_test.loc[:, 'instanceID'] = res.values
# # del data
# del data
# gc.collect()


##########################################################统计特征决赛用了clickTime之前所有天的统计，基本只用了平滑转化率特征，丢弃了点击数和转化数
#由于操作错误提交了包含creativeID_smooth和creativeID_rate两个特征的结果，后来丢掉rate效果会变差就一直留着了
#平滑user相关的特征特别废时间，初赛做过根据点击次数阈值来操作转化率，效果和平滑差不多但是阈值选择不太准
for feat_1 in ['creativeID','positionID','userID']:
    temp = pd.read_csv(temppath+'%s.csv' %feat_1)
    bs = BayesianSmoothing(1, 1)
    bs.update(temp[feat_1 + '_all'].values, temp[feat_1 + '_1'].values, 1000, 0.001)
    temp[feat_1 + '_smooth'] = (temp[feat_1 + '_1'] + bs.alpha) / (temp[feat_1 + '_all'] + bs.alpha + bs.beta)
    if feat_1 in ['creativeID']:
        temp[feat_1 + '_rate'] = temp[feat_1 + '_1'] / temp[feat_1 + '_all']
    temp.drop([feat_1 + '_1',feat_1 + '_all'],axis=1,inplace=True)
    X_loc_train = pd.merge(X_loc_train, temp, how='left', on=[feat_1, 'day'])
    X_loc_test = pd.merge(X_loc_test, temp, how='left', on=[feat_1, 'day'])
    del temp
    gc.collect()
    print(feat_1 + ' over...')
    X_loc_train.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
    X_loc_test.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
#类别少，不用平滑
for feat_1 in ['sitesetID']:
    temp = pd.read_csv(temppath+'%s.csv' %feat_1)
    temp[feat_1 + '_rate'] = temp[feat_1 + '_1'] / temp[feat_1 + '_all']
    X_loc_train = pd.merge(X_loc_train, temp, how='left', on=[feat_1, 'day'])
    X_loc_test = pd.merge(X_loc_test, temp, how='left', on=[feat_1, 'day'])
    del temp
    gc.collect()
    print(feat_1 + ' over...')
    X_loc_train.fillna(value=0, inplace=True)
    X_loc_test.fillna(value=0, inplace=True)

#三特征组合从周冠军分享的下载行为和网络条件限制，以及用户属性对app需求挖掘出
for feat_1,feat_2,feat_3 in[('appID','connectionType','positionID'),('appID','haveBaby','gender')]:
    temp = pd.read_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2+'_'+feat_3))
    bs = BayesianSmoothing(1, 1)
    bs.update(temp[feat_1+'_'+feat_2+'_'+feat_3 + '_all'].values, temp[feat_1+'_'+feat_2+'_'+feat_3 + '_1'].values, 1000, 0.001)
    temp[feat_1+'_'+feat_2+'_'+feat_3 + '_smooth'] = (temp[feat_1+'_'+feat_2+'_'+feat_3 + '_1'] + bs.alpha) / (temp[feat_1+'_'+feat_2+'_'+feat_3 + '_all'] + bs.alpha + bs.beta)
    temp.drop([feat_1+'_'+feat_2+'_'+feat_3+ '_1',feat_1+'_'+feat_2+'_'+feat_3 + '_all'],axis=1,inplace=True)
    X_loc_train = pd.merge(X_loc_train, temp, how='left', on=[feat_1,feat_2,feat_3, 'day'])
    X_loc_test = pd.merge(X_loc_test, temp, how='left', on=[feat_1,feat_2,feat_3, 'day'])
    del temp
    gc.collect()
    print(feat_1 + '_' + feat_2+'_'+feat_3+ ' over...')
    X_loc_train.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
    X_loc_test.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)

#userID和positionID的点击次数重要性排名靠前，所有统计特征只加了这一个点击次数
for feat_1,feat_2 in[('positionID','advertiserID'),('userID','sitesetID'),('positionID','connectionType'),('userID','positionID'),
                     ('appPlatform','positionType'),('advertiserID','connectionType'),('positionID','appCategory'),('appID','age'),
                     ('userID', 'appID'),('userID','connectionType'),('appCategory','connectionType'),('appID','hour'),('hour','age')]:
    temp = pd.read_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2))
    bs = BayesianSmoothing(1, 1)
    bs.update(temp[feat_1+'_'+feat_2 + '_all'].values, temp[feat_1+'_'+feat_2 + '_1'].values, 1000, 0.001)
    temp[feat_1+'_'+feat_2 + '_smooth'] = (temp[feat_1+'_'+feat_2 + '_1'] + bs.alpha) / (temp[feat_1+'_'+feat_2 + '_all'] + bs.alpha + bs.beta)
    if (feat_1,feat_2) in [('userID','positionID')]:
        temp.drop([feat_1 + '_' + feat_2 + '_1'], axis=1, inplace=True)
    else:
        temp.drop([feat_1+'_'+feat_2 + '_1',feat_1+'_'+feat_2 + '_all'],axis=1,inplace=True)
    X_loc_train = pd.merge(X_loc_train, temp, how='left', on=[feat_1,feat_2, 'day'])
    X_loc_test = pd.merge(X_loc_test, temp, how='left', on=[feat_1,feat_2, 'day'])
    del temp
    gc.collect()
    print(feat_1 + '_' + feat_2 + ' over...')
    X_loc_train.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)
    X_loc_test.fillna(value=bs.alpha/(bs.alpha + bs.beta), inplace=True)


##########################################################doTrick
X_loc_train=doTrick(X_loc_train)
X_loc_test=doTrick(X_loc_test)

##########################################################丢掉重要性低，缺失值多的原始特征
drop = ['hometown', 'haveBaby', 'telecomsOperator', 'userID', 'clickTime',
        'appPlatform', 'connectionType', 'marriageStatus', 'positionType',
        'gender', 'education', 'camgaignID', 'positionID','maybe_0'
        ]
X_loc_train.drop(drop, axis=1, inplace=True)
X_loc_train.fillna(value=0, inplace=True)
X_loc_test.drop(drop, axis=1, inplace=True)
X_loc_test.fillna(value=0, inplace=True)
print('over')
print(X_loc_train.shape)
print(X_loc_train.columns)
X_loc_train.to_csv(temppath+'2_smooth.csv',index=False)
X_loc_test.to_csv(temppath+'2_test_smooth.csv',index=False)

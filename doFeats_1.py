import numpy as np
import pandas as pd
import scipy as sp
import gc
import datetime
import random
import scipy.special as special

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
            print(self.alpha, self.beta)

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)

def readData(m_type='inner',drop=True):  ###################使用Trick时，left merge不改变顺序会比inner merge差1个万分点左右？
    X_train = pd.read_csv(rawpath+'train.csv')
    X_test = pd.read_csv(rawpath+'test.csv')
    if drop:
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

def doPre(data):
    data['day'] = data['clickTime'] // 1000000
    data['hour'] = data['clickTime'] % 1000000 // 10000
    return data

##########################################################installed文件关联user和app文件提取特征
userfile= pd.read_csv(rawpath+'user.csv')
appfile= pd.read_csv(rawpath+'app_categories.csv')
installed = pd.read_csv(rawpath+'user_installedapps.csv')
installed=installed.merge(userfile,how='left',on='userID')
installed=installed.merge(appfile,how='left',on='appID')

#app，appCat安装用户数
temp = installed.groupby('appID')['userID'].count().reset_index()
temp.columns=['appID','app_usercount']
temp.to_csv(iapath+'appInstalledusercount.csv',index=False)
temp = installed.groupby('appCategory')['userID'].count().reset_index()
temp.columns=['appCategory','appCat_usercount']
temp.to_csv(iapath+'appCatInstalledusercount.csv',index=False)

#user，edu，age,gender安装app数
temp = installed.groupby('userID')['appID'].count().reset_index()
temp.columns=['userID','user_appcount']
temp.to_csv(iapath+'userInstalledappscount.csv',index=False)
temp = installed.groupby('education')['appID'].count().reset_index()
temp.columns=['education','edu_appcount']
temp.to_csv(iapath+'eduuserInstalledappscount.csv',index=False)
temp = installed.groupby('age')['appID'].count().reset_index()
temp.columns=['age','age_appcount']
temp.to_csv(iapath+'ageuserInstalledappscount.csv',index=False)
temp = installed.groupby('gender')['appID'].count().reset_index()
temp.columns=['gender','gender_appcount']
temp.to_csv(iapath+'genderuserInstalledappscount.csv',index=False)
print('installed over...')

##########################################################actions文件提取特征，7天滑窗，统计用户安装的app数，app被安装的用户数
actions = pd.read_csv(rawpath+'user_app_actions.csv')
actions['day']=actions['installTime']//1000000
res=pd.DataFrame()
temp=actions[['userID','day','appID']]
for day in range(28,32):
    count=temp.groupby(['userID']).apply(lambda x: x['appID'][(x['day']<day).values & (x['day']>day-8).values].count()).reset_index(name='appcount')
    count['day']=day
    res=res.append(count,ignore_index=True)
res.to_csv(iapath+'all_user_seven_day_cnt.csv',index=False)
res=pd.DataFrame()
temp=actions[['userID','day','appID']]
for day in range(28,32):
    count=temp.groupby(['appID']).apply(lambda x: x['userID'][(x['day']<day).values & (x['day']>day-8).values].count()).reset_index(name='usercount')
    count['day']=day
    res=res.append(count,ignore_index=True)
res.to_csv(iapath+'all_app_seven_day_cnt.csv',index=False)
print('actions over...')




X_loc_train,X_loc_test=readData(m_type='inner',drop=True)
print('readData over')
X_loc_train=doPre(X_loc_train)
X_loc_test=doPre(X_loc_test)
print('doPre over...')

##########################################################统计特征，统计特征为点击数，转化数，转化率为转化数/点击数，
##########################################################初赛用7天滑窗算统计，决赛根据周冠军分享改为了使用了clickTime之前所有天算统计
for feat_1 in ['creativeID','positionID','userID','sitesetID']:
    gc.collect()
    res=pd.DataFrame()
    temp=X_loc_train[[feat_1,'day','label']]
    for day in range(28,32):
        count=temp.groupby([feat_1]).apply(lambda x: x['label'][(x['day']<day).values].count()).reset_index(name=feat_1+'_all')
        count1=temp.groupby([feat_1]).apply(lambda x: x['label'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_1')
        count[feat_1+'_1']=count1[feat_1+'_1']
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,' over')
    res.to_csv(temppath+'%s.csv' %feat_1, index=False)


for feat_1,feat_2 in[('positionID','advertiserID'),('userID','sitesetID'),('positionID','connectionType'),('userID','positionID'),
                     ('appPlatform','positionType'),('advertiserID','connectionType'),('positionID','appCategory'),('appID','age'),
                     ('userID', 'appID'),('userID','connectionType'),('appCategory','connectionType'),('appID','hour'),('hour','age')]:
    gc.collect()
    res=pd.DataFrame()
    temp=X_loc_train[[feat_1,feat_2,'day','label']]
    for day in range(28,32):
        count=temp.groupby([feat_1,feat_2]).apply(lambda x: x['label'][(x['day']<day).values].count()).reset_index(name=feat_1+'_'+feat_2+'_all')
        count1=temp.groupby([feat_1,feat_2]).apply(lambda x: x['label'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_1')
        count[feat_1+'_'+feat_2+'_1']=count1[feat_1+'_'+feat_2+'_1']
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,feat_2,' over')
    res.to_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2), index=False)

for feat_1,feat_2,feat_3 in[('appID','connectionType','positionID'),('appID','haveBaby','gender')]:
    gc.collect()
    res=pd.DataFrame()
    temp=X_loc_train[[feat_1,feat_2,feat_3,'day','label']]
    for day in range(28,32):
        count=temp.groupby([feat_1,feat_2,feat_3]).apply(lambda x: x['label'][(x['day']<day).values].count()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_all')
        count1=temp.groupby([feat_1,feat_2,feat_3]).apply(lambda x: x['label'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_1')
        count[feat_1+'_'+feat_2+'_'+feat_3+'_1']=count1[feat_1+'_'+feat_2+'_'+feat_3+'_1']
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,feat_2,feat_3,' over')
    res.to_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2+'_'+feat_3), index=False)


##########################################################比赛官方群里大神分享过的，这里用app平均回流时间做特征，缺失的用app类别的平均回流时间替代
X_loc_train,X_loc_test=readData(m_type='inner',drop=False)
del X_loc_test
X_loc_train=X_loc_train.loc[X_loc_train['label']==1,:]
X_loc_train['cov_diffTime']=X_loc_train['conversionTime']-X_loc_train['clickTime']
grouped=X_loc_train.groupby('appID')['cov_diffTime'].mean().reset_index()
grouped.columns=['appID','cov_diffTime']
grouped.to_csv(temppath+'app_cov_diffTime.csv',index=False)

grouped=X_loc_train.groupby('appCategory')['cov_diffTime'].mean().reset_index()
grouped.columns=['appCategory','appCat_cov_diffTime']
grouped.to_csv(temppath+'appCat_cov_diffTime.csv',index=False)



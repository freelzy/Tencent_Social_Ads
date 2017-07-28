import pandas as pd
import numpy as np
import math

def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)
    newData=dataMat-meanVal
    return newData,meanVal

newData,meanVal=zeroMean(X_train)
res=newData.corr(method='pearson')

def f(x):
    res=1/(1+np.e**(-x))
    return res

def f_ver(x):
    res=np.log(x/(1-x))
    return res


df=pd.read_csv(r'a.csv')
print(df.prob.mean())
avg=0.027232030146226882#0.0273
b=[-2,2]
df.prob1=df.prob
while abs(np.mean(df.prob1)-avg)>0.00001:
    mid=(b[0]+b[1])/2.0
    df.prob1=df.prob.apply(lambda x:math.log(x/(1-x)))
    df.prob1=df.prob1.apply(lambda x:x+mid)
    df.prob1=df.prob1.apply(lambda x:1/(1+math.exp(-x)))
    if np.mean(df.prob1)>avg:
        b[1]=mid
    else:
        b[0]=mid
df.prob=df.prob1
del df.prob1
df.to_csv(r'submission.csv',index=False)
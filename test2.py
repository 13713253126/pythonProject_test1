import sklearn

import os
print(os.path.abspath('.'))

import pandas as pd
import numpy as np
from pandas import DataFrame
# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")

# 导入python绘图matplotlib
import matplotlib.pyplot as plt



# 设置绘图大小
plt.style.use({'figure.figsize':(25,20)})

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

#es.search(index='properties_weatherstation_2022-04', filter_path=filter_path, body=body)  # 指定查询条件

import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000

from functools import reduce
import datetime
#定义一个时间戳转换函数
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import explained_variance_score,mean_absolute_error, mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
from distfit import distfit
from flask import Flask, redirect, url_for, request
#参数一：表？
df_final=pd.read_csv('/home/dyn/change/weatherstation-leftjoin-202204.csv')

import datetime
#定义一个时间戳转换函数
import time
def time2stamp(cmnttime):   #转时间戳函数
    cmnttime_after=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(float(cmnttime)/1000))
    return cmnttime_after
df_final['timestamp']=df_final['timestamp'].apply(time2stamp)
print(df_final.head(5))
print(df_final.tail(5))

#取变量列名
valuename = list(df_final.iloc[:,2:])
valuename
print(df_final.info())
#循环删除为文本的列里是'NaN'的
for i in valuename:
    #删除NAN
    #object
    if df_final[i].dtypes == 'object':
        print(df_final[i].dtypes)
        #删除NAN
        index=df_final[df_final[i]=='NaN'].index
        if len(index)>0:
            print(len(index))
            df_final=df_final.drop(index=df_final[df_final[i].isna()].index)

#对变量列转小数
#valuename = ['humidity','windForce','windSpeed','temperature']
for i in valuename:
    #加一步如果列为纯文本列，如风向，则不转
    #value转浮点
    df_final[i]=pd.to_numeric(df_final[i])
print(df_final.info())

#建索引
df_final['data'] = pd.to_datetime(df_final['timestamp'])
df_final = df_final.set_index('data')
df_final.head(5)

#循环删除有空值行行
for i in valuename:
    #删除空行
    index=df_final[df_final[i].isna()].index
    if len(index)>0:
        df_final=df_final.drop(index=df_final[df_final[i].isna()].index)
#df_final=df_final.drop(index=df_final[df_final['windSpeed'].isna()].index)


#相关系数
print(df_final.corr())

df_final.describe()
#参数二：target
#分目标值与变量
target_value='humidity'
target=df_final[[target_value]]
data_value=list(df_final.iloc[:,2:])
data_value.remove(target_value)
data= df_final[data_value]
print(target.head(5))
print(data.head(5))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,target,random_state=22)
#数据集划分
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

begindata = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(begindata)


from sklearn.ensemble import RandomForestRegressor
estimator=RandomForestRegressor()
#加入网格搜索，交叉验证
param_dict={"max_depth":[3,5,7],"n_estimators":[5,10,20,50,100,200]}
# n_estimators：决策树的个数 max_depth：最大树深，树太深会造成过拟合
#对树深，树颗数进行网格搜索
estimator=GridSearchCV(estimator,param_grid=param_dict,cv=3)
#三折交叉验证
estimator.fit(x_train,y_train)
#随机森林模型
y_predict=estimator.predict(x_test)
#print("真实值与预测值",y_test==y_predict)
accuracy=estimator.score(x_test,y_test)
print("准确率",accuracy)
print("最佳参数",estimator.best_params_)
print("最佳结果",estimator.best_score_)



enddata = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(enddata)

predict1=estimator.predict([[0,0,24]])
print(predict1)

#import joblib
#joblib.dump(estimator, '/home/dyn/dbz/python sklearn model/sjsl01.pkl')

#参数三： 滑动频率
freq= '1h'
df0 = df_final[[target_value]]
df0 = df0[~df0.index.duplicated()]
#df0=df0.resample('1d', label='left',closed='left').interpolate('linear') # 线性插值
df0=df0.resample(freq, label='left',closed='left').max().ffill() #最大值聚合 为空的找前一个元素填充
df0.head()

begindata = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(begindata)

# 对模型p,q进行定阶
warnings.filterwarnings("ignore")  # specify to ignore warning messages
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

pmax = int(2)
qmax = int(1)
bic_matrix = []
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        # 存在部分报错，所以用try来跳过报错。
        try:
            tmp.append(sm.tsa.arima.ARIMA(df0, order=(p, 1, q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)

# 从中可以找出最小值

bic_matrix = pd.DataFrame(bic_matrix)

# 先用stack展平，然后用idxmin找出最小值位置。

p, q = bic_matrix.stack().astype(float).idxmin()

print(u'BIC最小的p值为：%s' % (p))
print(u'BIC最小的q值为：%s' % (q))
# 取BIC信息量达到最小的模型阶数，结果p为0，q为1，定阶完成。

enddata = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(enddata)

m1 = pd.date_range('2022-04-01 00:00:00', '2022-04-01 23:59:59', freq=freq)
m = len(m1)
if m < 2:
    m = 2

m

from statsmodels.tsa.statespace.sarimax import SARIMAX
# fit model
model = SARIMAX(df0, order=(p,1,q), seasonal_order=(0, 1, 1, m)).fit(disp=-1)
# make prediction
#yhat = model_fit.predict(len(data1), len(data1))
#data1
ARIMA_predict=model.predict('2022-04-30 18:00:00','2022-05-05 00:00:00')

print(ARIMA_predict)
#import joblib
#joblib.dump(model, '/home/dyn/dbz/python sklearn model/SARIMAX01.pkl')

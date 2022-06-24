import sklearn

import os
print(os.path.abspath('.'))

import pandas as pd
import numpy as np
from pandas import DataFrame
# 忽略烦人的红色提示
import warnings
#warnings.filterwarnings("ignore")

# 导入python绘图matplotlib
import matplotlib.pyplot as plt

# 使用ipython的魔法方法，将绘制出的图像直接嵌入在notebook单元格中
#%matplotlib inline

# 设置绘图大小
plt.style.use({'figure.figsize':(25,20)})

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

#es.search(index='properties_weatherstation_2022-04', filter_path=filter_path, body=body)  # 指定查询条件

import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000

#参数一：表？
df_final=pd.read_csv('/home/dyn/change/weatherstation-leftjoin-202204.csv')

import datetime
#定义一个时间戳转换函数
import time
def time2stamp(cmnttime):   #转时间戳函数
    cmnttime_after=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(float(cmnttime)/1000))
    return cmnttime_after
df_final['timestamp']=df_final['timestamp'].apply(time2stamp)

import streamlit as st
import numpy as np
import pandas as pd
st.title("随机森林/回归")
st.write("hello")
st.write(df_final.head(5))

df_final_value_list=list(df_final)
df_final_value_list.remove('Unnamed: 0')
df_final_value_list.remove('timestamp')
#df_final_value_list

target_value=st.selectbox('选择目标值',df_final_value_list)
st.write('目标值:',target_value)

df_final_value_list2=list(df_final)
df_final_value_list2.remove('Unnamed: 0')
df_final_value_list2.remove('timestamp')
df_final_value_list2.remove(target_value)

data_value=st.multiselect('选择特征值',df_final_value_list2)
st.write('特征值:',data_value)

target_value_mid = [target_value]
valuename = target_value_mid + data_value
st.write('valuename:',valuename)

# 循环删除为文本的列里是'NaN'的
for i in valuename:
    # 删除NAN
    # object
    if df_final[i].dtypes == 'object':
        print(df_final[i].dtypes)
        # 删除NAN
        index = df_final[df_final[i] == 'NaN'].index
        if len(index) > 0:
            print(len(index))
            df_final = df_final.drop(index=df_final[df_final[i].isna()].index)

# 对变量列转小数
# valuename = ['humidity','windForce','windSpeed','temperature']
for i in valuename:
    # 加一步如果列为纯文本列，如风向，则不转
    if i != 'windDirection':
        # value转浮点
        df_final[i] = pd.to_numeric(df_final[i])

# 建索引
df_final['data'] = pd.to_datetime(df_final['timestamp'])
df_final = df_final.set_index('data')

# 循环删除有空值行行
for i in valuename:
    # 删除空行
    index = df_final[df_final[i].isna()].index
    if len(index) > 0:
        df_final = df_final.drop(index=df_final[df_final[i].isna()].index)
# df_final=df_final.drop(index=df_final[df_final['windSpeed'].isna()].index)


st.write(df_final.head(5))

# 参数二：target
# 分目标值与变量
target = df_final[[target_value]]
data = df_final[data_value]

st.write(target.head(5))
st.write(data.head(5))

if st.button('开始训练'):

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=22)
    # 数据集划分
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV



    begindata = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    st.write('开始时间:',begindata)

    from sklearn.ensemble import RandomForestRegressor
    estimator = RandomForestRegressor()
    # 加入网格搜索，交叉验证
    param_dict = {"max_depth": [3, 5, 7], "n_estimators": [5, 10, 20, 50, 100, 200]}
    # n_estimators：决策树的个数 max_depth：最大树深，树太深会造成过拟合
    # 对树深，树颗数进行网格搜索
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    #   三折交叉验证
    estimator.fit(x_train, y_train)
    # 随机森林模型
    y_predict = estimator.predict(x_test)
    # print("真实值与预测值",y_test==y_predict)
    accuracy = estimator.score(x_test, y_test)
    # print("准确率",accuracy)
    # print("最佳参数",estimator.best_params_)
    # print("最佳结果",estimator.best_score_)

    enddata = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    st.write('结束时间:',enddata)
    st.write('准确率:',accuracy)
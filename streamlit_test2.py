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
st.title("time_ARIMA")
st.write("hello")
st.write(df_final.head(5))

df_final_value_list=list(df_final)
df_final_value_list.remove('Unnamed: 0')
df_final_value_list.remove('timestamp')
#df_final_value_list

target_value=st.selectbox('选择目标值',df_final_value_list)
st.write('目标值:',target_value)

target_value_mid = [target_value]
valuename = target_value_mid

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


#st.write(df_final.head(5))

list_freq=['1h','12h','1d']
list_type=['最大值','最小值','平均值']

freq=st.selectbox('选择滑动窗口',list_freq)
freq_type=st.selectbox('选择聚合模式',list_type)
st.write('滑动窗口:',freq)
st.write('选择聚合模式:',freq_type)

df0 = df_final[[target_value]]
df0 = df0[~df0.index.duplicated()]

if freq_type=='最大值':
    df0=df0.resample(freq, label='left',closed='left').max().ffill() #最大值聚合 为空的找前一个元素填充
if freq_type=='最小值':
    df0=df0.resample(freq, label='left',closed='left').max().ffill() #最大值聚合 为空的找前一个元素填充
if freq_type=='平均值':
    df0=df0.resample(freq, label='left',closed='left').max().ffill() #最大值聚合 为空的找前一个元素填充
#st.write(df0.head(5))



if st.button('开始训练'):

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

    st.write('BIC最小的p值为：',p)
    st.write('BIC最小的q值为：',q)
    #print(u'BIC最小的p值为：%s' % (p))
    #print(u'BIC最小的q值为：%s' % (q))
    # 取BIC信息量达到最小的模型阶数，结果p为0，q为1，定阶完成。


    m1 = pd.date_range('2022-04-01 00:00:00', '2022-04-01 23:59:59', freq=freq)
    m = len(m1)
    if m < 2:
        m = 2

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # fit model
    model = SARIMAX(df0, order=(p,1,q), seasonal_order=(0, 1, 1, m)).fit(disp=-1)
    # make prediction
    #yhat = model_fit.predict(len(data1), len(data1))
    #data1
    st.write('训练结束')

    #ARIMA_predict=model.predict('2022-04-30 18:00:00','2022-05-05 00:00:00')

    #st.write(ARIMA_predict)

id1=st.text_input('预测开始时间')
id2=st.text_input('预测结束时间')

if st.button('预测'):
    time1 = pd.to_datetime(id1)
    time2 = pd.to_datetime(id2)
    predict2 = model.predict(time1, time2)
    st.write(predict2)
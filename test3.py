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
import matplotlib as mpl
from functools import reduce
import joblib

# 参数一：表？
# df_final=pd.read_csv('D:\estemp\weatherstation-leftjoin-202204.csv')

from flask import Flask, redirect, url_for, request

app = Flask(__name__)


@app.route('/api/')
def hello_name():
    # 参数一：表？
    df_final = pd.read_csv('/home/dyn/change/weatherstation-leftjoin-202204.csv')
    id0 = request.args.get('id0')  # 模型名称
    id1 = request.args.get('id1')  # 设备id
    modelhead = '/home/dyn/dbz/python sklearn model/'
    tail = '.pkl'
    modelname = modelhead + id0 + '-' + id1 + tail  # 模型保存路径拼接

    target_value = request.args.get('id2')  # 目标值
    data_value_mid = request.args.get('id3')  # 特征值传参
    data_value = data_value_mid.split(',')
    data_value = list(data_value)  # 特征值
    data_value_count = len(data_value)  # 特征值个数

    import datetime
    # 定义一个时间戳转换函数
    import time
    def time2stamp(cmnttime):  # 转时间戳函数
        cmnttime_after = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(cmnttime) / 1000))
        return cmnttime_after

    df_final['timestamp'] = df_final['timestamp'].apply(time2stamp)

    target_value_mid = [target_value]
    valuename = target_value_mid + data_value

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

    # 参数二：target
    # 分目标值与变量
    target = df_final[[target_value]]
    data = df_final[data_value]

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=22)
    # 数据集划分
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV

    begindata = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # print(begindata)

    from sklearn.ensemble import RandomForestRegressor
    estimator = RandomForestRegressor()
    # 加入网格搜索，交叉验证
    param_dict = {"max_depth": [3, 5, 7], "n_estimators": [5, 10, 20, 50, 100, 200]}
    # n_estimators：决策树的个数 max_depth：最大树深，树太深会造成过拟合
    # 对树深，树颗数进行网格搜索
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    # 三折交叉验证
    estimator.fit(x_train, y_train)
    # 随机森林模型
    y_predict = estimator.predict(x_test)
    # print("真实值与预测值",y_test==y_predict)
    accuracy = estimator.score(x_test, y_test)
    # print("准确率",accuracy)
    # print("最佳参数",estimator.best_params_)
    # print("最佳结果",estimator.best_score_)

    enddata = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # print(enddata)

    joblib.dump(estimator, modelname)

    return '开始时间:' + str(begindata) + '准确率:' + str(accuracy) + '结束时间:' + str(enddata)


print('1000')
# http://192.168.108.181:8001/api/?id0=随机森林&id1=123456&id2=humidity&id3=windForce,windSpeed,temperature

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001)

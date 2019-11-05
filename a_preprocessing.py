import gc
import os

# import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

global train_data
global test_data
test_data = pd.read_csv(
    './data/test_raw.csv')
train_data = pd.read_csv(
    './data/train_raw.csv')


def get_preprocessed_data():
    for col in train_data.columns:
        if train_data[col].isnull().any():
            print(col)

    for col in test_data.columns:
        if test_data[col].isnull().any():
            print(col)
    if(os.path.exists('./results/test_data.csv')):
        print("数据预处理完毕")
    else:

        print("正在进行数据预处理")
        train_data['bankCard'].fillna(-999, inplace=True)
        test_data['bankCard'].fillna(-999, inplace=True)

        train_data.bankCard = train_data.bankCard.astype('int64')
        train_data.certValidStop = train_data.certValidStop.astype('int64')

        test_data.bankCard = test_data.bankCard.astype('int64')
        test_data.certValidStop = test_data.certValidStop.astype('int64')

        # 发现certValidStop异常值，新增特征stop是否异常
        train_data['is_InValidStop'] = train_data.certValidStop.apply(
            lambda x: str(x).startswith('2'))
        test_data['is_InValidStop'] = test_data.certValidStop.apply(
            lambda x: str(x).startswith('2'))

        # 将异常值变为中位数、众数
        median = train_data['certValidStop'].mode()

        train_data.certValidStop = train_data.certValidStop.replace(
            256000000000, median)
        
        median = test_data['certValidStop'].mode()
        test_data.certValidStop = test_data.certValidStop.replace(
            256000000000, median)
        print("数据预处理完成")

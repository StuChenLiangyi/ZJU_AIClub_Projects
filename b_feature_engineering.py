import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import nltk
import os
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import gc
from a_preprocessing import train_data, test_data

global data, train, test, labels, label_0_df, label_1_df

data = train = test = category_column = None
labels = pd.read_csv(
    './data/train_target.csv')
label_0_df = labels[labels.target == 0]
label_1_df = labels[labels.target == 1]

def make_features(using_raw_features=False):  # 制造新特征
    global train_data, test_data, data, category_column
    if(os.path.exists('./results/test_data.csv')):
        test_data = pd.read_csv('./results/test_data.csv')
        train_data = pd.read_csv('./results/train_data.csv')
        print("检测到数据已完成生成新特征")
    else: 
        print("正在生成新特征")
        print(train_data.shape)
        # 统计信息缺失数量,得到信息缺失度特征
        nums = []
        for row in train_data.itertuples():
            n = 0
            for col in row:
                if(str(col) == '-999'):
                    n += 1
            nums.append(n)
        train_data['missing_columns'] = nums

        nums = []
        for row in test_data.itertuples():
            n = 0
            for col in row:
                if(str(col) == '-999'):
                    n += 1
            nums.append(n)
        test_data['missing_columns'] = nums

        train_data.index = train_data.id
        train_1_df = train_data.loc[label_1_df.index]
        train_0_df = train_data.loc[label_0_df.index]
        # train_0_df.head(2)

        # 对证件号的处理,转变为违约次数特征、历史违约率,在test中出现，但train中未出现的取train均值
        fd = nltk.FreqDist(train_1_df.certId)
        train_data['disobey_num'] = train_data.certId.apply(lambda x: fd[x])
        uniques=train_data.certId.unique()
        add=0
        for x in uniques:
            add+=fd[x]
        average=1.0*add/len(uniques)
        test_data['disobey_num'] = test_data['certId'].apply(lambda x: fd[x] if x in uniques else average)
        train_1_df['disobey_num'] = train_1_df['certId'].apply(lambda x: fd[x])
        train_0_df['disobey_num'] = train_0_df['certId'].apply(lambda x: fd[x])

        # 对证件号的处理,转变为违约率特征
        fd1 = nltk.FreqDist(train_data.certId)
        train_data['disobey_rate'] = train_data['certId'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))
        
        add=0
        for x in uniques:
            add+=round(fd[x]/(0.000000000001+fd1[x]), 4)
        average=1.0*add/len(uniques)
        
        test_data['disobey_rate'] = test_data['certId'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4) if x in uniques else average )
        train_1_df['disobey_rate'] = train_1_df['certId'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))
        train_0_df['disobey_rate'] = train_0_df['certId'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))

        
        # 居住地取前4位
        train_data['residentAddr'] = train_data['residentAddr'].apply(lambda x: str(x)[
                                                              :4])
        test_data['residentAddr'] = test_data['residentAddr'].apply(lambda x: str(x)[
                                                                    :4])
        train_0_df['residentAddr'] = train_0_df['residentAddr'].apply(lambda x: str(x)[
                                                                      :4])
        train_1_df['residentAddr'] = train_1_df['residentAddr'].apply(lambda x: str(x)[
                                                                      :4])

        # 继续对地区进行分析,将dist转化为只保留前4位,缩小粒度，疑似指代省份
        train_data['dist'] = train_data['dist'].apply(lambda x: str(x)[:4])
        test_data['dist'] = test_data['dist'].apply(lambda x: str(x)[:4])
        train_0_df['dist'] = train_0_df['dist'].apply(lambda x: str(x)[:4])
        train_1_df['dist'] = train_1_df['dist'].apply(lambda x: str(x)[:4])
        
        # 对地区的处理,新增特征：所处dist违约率
        fd = nltk.FreqDist(train_1_df.dist)
        fd1 = nltk.FreqDist(train_data.dist)
        uniques=train_data.dist.unique()
        add=0
        for x in uniques:
            add+=round(fd[x]/(0.000000000001+fd1[x]), 4)
        average=1.0*add/len(uniques)
        train_data['dist_disobey_rate'] = train_data['dist'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))
        test_data['dist_disobey_rate'] = test_data['dist'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4) if x in uniques else average)
        train_1_df['dist_disobey_rate'] = train_1_df['dist'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))
        train_0_df['dist_disobey_rate'] = train_0_df['dist'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))

        # 对银行卡号分析,将bankCard转化为只保留前6位,缩小粒度
        train_data['bankCard'] = train_data['bankCard'].apply(
            lambda x: str(x)[:6])
        test_data['bankCard'] = test_data['bankCard'].apply(
            lambda x: str(x)[:6])
        train_0_df['bankCard'] = train_0_df['bankCard'].apply(
            lambda x: str(x)[:6])
        train_1_df['bankCard'] = train_1_df['bankCard'].apply(
            lambda x: str(x)[:6])

        # 对卡号的处理,新增特征：银行卡号违约次数
        fd = nltk.FreqDist(train_1_df.bankCard)
        fd1 = nltk.FreqDist(train_data.bankCard)
        
        uniques=train_data.bankCard.unique()
        add=0
        for x in uniques:
            add+=fd[x]
        average=1.0*add/len(uniques)
        
        train_data['card_disobey_num'] = train_data['bankCard'].apply(
            lambda x: fd[x])
        test_data['card_disobey_num'] = test_data['bankCard'].apply(
            lambda x: fd[x] if x in uniques else average)
        train_1_df['card_disobey_num'] = train_1_df['bankCard'].apply(
            lambda x: fd[x])
        train_0_df['card_disobey_num'] = train_0_df['bankCard'].apply(
            lambda x: fd[x])

        # 对卡号的处理,新增特征：银行卡号违约率
        add=0
        for x in uniques:
            add+=round(fd[x]/(0.000000000001+fd1[x]), 4)
        average=1.0*add/len(uniques)
        
        train_data['card_disobey_rate'] = train_data['bankCard'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))
        test_data['card_disobey_rate'] = test_data['bankCard'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4) if x in uniques else average)
        train_1_df['card_disobey_rate'] = train_1_df['bankCard'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))
        train_0_df['card_disobey_rate'] = train_0_df['bankCard'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))

        # 对居住地分析，新增 居住地和dist是否一致
        train_data['is_same_addr'] = (
            train_data.residentAddr == train_data.dist)
        test_data['is_same_addr'] = (test_data.residentAddr == test_data.dist)
        train_0_df['is_same_addr'] = (
            train_0_df.residentAddr == train_0_df.dist)
        train_1_df['is_same_addr'] = (
            train_1_df.residentAddr == train_1_df.dist)

        # 新增居住地违约率特征
        fd = nltk.FreqDist(train_1_df.dist)
        fd1 = nltk.FreqDist(train_data.dist)
        uniques=train_data.residentAddr.unique()
        add=0
        for x in uniques:
            add+=round(fd[x]/(0.000000000001+fd1[x]), 4)
        average=1.0*add/len(uniques)
        
        train_data['addr_disobey_rate'] = train_data['residentAddr'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))
        test_data['addr_disobey_rate'] = test_data['residentAddr'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4) if x in uniques else average)
        train_1_df['addr_disobey_rate'] = train_1_df['residentAddr'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))
        train_0_df['addr_disobey_rate'] = train_0_df['residentAddr'].apply(
            lambda x: round(fd[x]/(0.000000000001+fd1[x]), 4))

        # 新增特征，省份是否一致
        train_data['is_same_prov'] = (
            train_data.residentAddr == train_data.dist)
        test_data['is_same_prov'] = (test_data.residentAddr == test_data.dist)
        train_0_df['is_same_prov'] = (
            train_0_df.residentAddr == train_0_df.dist)
        train_1_df['is_same_prov'] = (
            train_1_df.residentAddr == train_1_df.dist)
        
    #发现证件日期会使训练集、测试集产生差异，去除。
        # 有效期发现均可整/60/60/24/30，转为月数
        train_data['certValidBegin'] = train_data['certValidBegin'].apply(
            lambda x: round(x/60/60/24/30, 0))
        train_data['certValidStop'] = train_data['certValidStop'].apply(
            lambda x: round(x/60/60/24/30, 0))

        test_data['certValidBegin'] = test_data['certValidBegin'].apply(
            lambda x: round(x/60/60/24/30, 0))
        test_data['certValidStop'] = test_data['certValidStop'].apply(
            lambda x: round(x/60/60/24/30, 0))

        # 新增证件有效期间隔月数
        train_data['certValidMonths'] = train_data['certValidStop'] - \
            train_data['certValidBegin']
        test_data['certValidMonths'] = test_data['certValidStop'] - \
            train_data['certValidBegin']

#         # 新增证件有效期间隔年数,作为类别
#         train_data['certValidYears'] = train_data['certValidMonths'].apply(
#             lambda x: round(x/12, 0))
#         test_data['certValidYears'] = test_data['certValidMonths'].apply(
#             lambda x: round(x/12, 0))

        if(using_raw_features):
            drop_list = [
                     'certValidMonths','is_InValidStop',
                     'missing_columns',
                     'disobey_num',
                     'disobey_rate',
                     'dist_disobey_rate',
                     'card_disobey_num',
                     'card_disobey_rate',
                     'is_same_addr',
                     'addr_disobey_rate',
                     'is_same_prov']
        else:
            drop_list = []
        
        train_data = train_data.drop(drop_list, axis=1)
        test_data = test_data.drop(drop_list, axis=1)

        train_data.to_csv('./results/train_data.csv', index=0)
        test_data.to_csv('./results/test_data.csv', index=0)
        print("完成特征生成")
        

           


def one_hot_data():
    # 对所有非数值型数据做独热编码
    global data, test, train
    if(os.path.exists('./results/test.csv')):
        test = pd.read_csv('./results/test.csv')
        train = pd.read_csv('./results/train.csv')
        train.index = train.id
        test.index = test.id
        print(train.shape)
        print("检测到已完成独热编码")

    else:
        print("正在进行独热编码")
#         category_column = ['bankCard','ethnic','residentAddr','dist','job']  # 类别全集，其余全部当做数值型
        category_column =["gender","job", "loanProduct",
                          "basicLevel","ethnic"]
        number_column = ['age', 'lmt','residentAddr','dist','bankCard']
#         number_column =['age', 'lmt']
#         for col in train_data.columns:
#             if col not in number_column:
#                 n = list(train_data[col].unique())
#                 m = list(test_data[col].unique())
#                 if(len(set(m+n)) < 100):
#                     category_column.append(col)
#                 else:
#                     number_column.append(col)

        # category_column.remove('certValidBegin')
        # category_column.remove('certValidStop')
        # 将年龄\基础评级\个数\月数\率作为数值型

        data = pd.concat([train_data, test_data], ignore_index=True)
        for col in category_column:
            data[col] = data[col].astype('category')
        for col in number_column:
            data[col] = data[col].astype('float64') 
        
        # 数值型数据缺失用众数填充,非类别型将缺失看为一种类型
        for col in data.columns:
            if col not in (category_column):
                fd = nltk.FreqDist(data[col])
                data[col] = data[col].replace(-999, fd.max())
        #         print(col,'数值，众数填充')

#         #对所有数值型正态
#         cols = data.select_dtypes(exclude=['category']).columns
#         print("正态")
# #         print(cols)
#         for col in cols:
#             if col != 'id':
#                 data[col]=np.log1p(data[col])

        print ("cut")
        # 等频／等距／卡方等
        data["age_bin"] = pd.cut(data["age"],20,labels=False)
        data = data.drop(['age'], axis=1)
        data["dist_bin"] = pd.qcut(data["dist"],60,labels=False)
        data = data.drop(['dist'], axis=1)
        data["lmt_bin"] = pd.qcut(data["lmt"],50,labels=False)
        data = data.drop(['lmt'], axis=1)
        data["setupHour_bin"] = pd.qcut(data["setupHour"],10,labels=False)
        data = data.drop(['setupHour'], axis=1)
        
        #独热编码
        cols = data.select_dtypes(include=['category']).columns
        print(cols)
        data = pd.get_dummies(data, columns=cols)

        train = data[data.id < 132030]
        test = data[data.id >= 132030]

        train.to_csv('./results/train.csv', index=0)
        test.to_csv('./results/test.csv', index=0)
        print("独热编码完成,训练集维度为",train.shape)
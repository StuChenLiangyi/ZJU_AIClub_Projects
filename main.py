import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import nltk

# import xgboost as xgb
# from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,accuracy_score

import a_preprocessing
import b_feature_engineering
import c_data_sample
import os
from sklearn.externals import joblib
from d_xgb import get_trained_xgb, get_model_metrics
from d_catboost import get_trained_ctb

from c_data_sample import eliminate_different_data, make_new_sample
from e_feature_combination import get_mul_features, get_add_features, apply_combine_features
import gc
import b_feature_engineering
import a_preprocessing
from f_output import output

def start(using_raw_features=True, nums=1000, scale_pos_weight=130, neighbor_nums=130000):
    gc.collect()
    a_preprocessing.get_preprocessed_data()  # 预处理
    b_feature_engineering.make_features(
        using_raw_features=using_raw_features)  # 新建特征
    # print("使用原始特征：{0},使用模型：{1},是否独热编码:{2},是否采用近邻点:{3},是否采用人工生成样本:{4}".format(
    #     using_raw_features, 'ctb', 'no', 'no', 'no'))
    # ctb = get_trained_ctb(b_feature_engineering.train_data,
    #                       b_feature_engineering.labels.target, scale_pos_weight)  # catboost结果
    # output(ctb,b_feature_engineering.test_data,
    #        './results/submission_ctb_0.csv')
    b_feature_engineering.one_hot_data()

    make_new_sample(b_feature_engineering.train,
                    b_feature_engineering.labels.target,
                    nums=nums)  # 生成新样本数据

    eliminate_different_data(neighbor_nums=neighbor_nums)

    print("使用原始特征：{0},使用模型：{1},是否独热编码:{2},是否采用近邻点:{3},是否采用人工生成样本:{4}".format(
        using_raw_features, 'ctb', 'yes', 'yes', 'yes'))

    ytrain = pd.merge(c_data_sample.train_selected,
                      b_feature_engineering.labels, on='id').target

    print(c_data_sample.train_selected.shape, len(ytrain))
    ctb = get_trained_ctb(c_data_sample.train_selected,
                          ytrain.values, scale_pos_weight)  # catboost结果
    output(ctb,b_feature_engineering.test,'./results/submission_ctb_1.csv')
    # return
    if(os.path.exists('./results/clf_0.model')):
        clf_0 = joblib.load('./results/clf_0.model')
        print("检测到初始模型已生成")
    else:

        # 1：0正负例样本采样，用于调节不平衡
        x_train, scale_pos_weight = c_data_sample.get_sampled_data(
            c_data_sample.train_selected, 0.95, 0.95)  # 0.8，0.75建完测试时
        # 在基本特征上得到第一个初始模型
        clf_0 = get_trained_xgb(x_train, scale_pos_weight)
        joblib.dump(clf_0, './results/clf_0.model')
        get_model_metrics(clf_0, x_train)

        del x_train
        gc.collect()
    # x_train, scale_pos_weight = c_data_sample.get_sampled_data(
    #         c_data_sample.train_selected, 0.9, 0.95)
    # get_model_metrics(clf_0,x_train)
    # del x_train
    # gc.collect()
    # 得到基础模型特征重要性排序
    weight_kv_0 = clf_0.get_booster().get_score()
    top_7 = list(weight_kv_0.keys())[:7]
    print(list(weight_kv_0.keys())[:20])

    # 得到top和其他列组合的加减法特征
    get_mul_features(top_7)
    get_add_features(top_7)
    apply_combine_features()


if __name__ == '__main__':
    start()

# import numpy as np
# import pandas as pd
# # import matplotlib.pyplot as plt
# import nltk

# import xgboost as xgb
# from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,accuracy_score
import a_preprocessing
import b_feature_engineering
import c_data_sample
import os
from sklearn.externals import joblib
from d_xgb import get_trained_xgb, get_model_metrics
from c_data_sample import eliminate_different_data
from e_feature_combination import get_mul_features, get_add_features,apply_combine_features
import gc

def main():
    gc.collect() 
    a_preprocessing.get_preprocessed_data()
    b_feature_engineering.make_features()
    b_feature_engineering.one_hot_data()
    eliminate_different_data()
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
        get_model_metrics(clf_0,x_train)

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
    main()

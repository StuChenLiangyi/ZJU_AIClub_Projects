import nltk
import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import b_feature_engineering
from b_feature_engineering import labels, label_0_df, label_1_df
import gc
import c_data_sample
def get_trained_xgb(x_train, scale_pos_weight):
    # 训练xgb模型
    print("开始训练模型")
    print(x_train.shape)
    print(datetime.now().strftime('%Y-%m-%d %H:%M'))
    clf = xgb.XGBClassifier(max_depth=8, n_estimators=200,
                            scale_pos_weight=scale_pos_weight/4,
                            eval_metric='auc',
                            objective='binary:logistic',
                            # colsample_bytree=0.8,
                            # subsample=0.8,
                            nthread=10,
                            learning_rate=0.1
                            )
    y_train=pd.merge(x_train, labels, on='id').target
    print(dict(nltk.FreqDist(list(y_train))))
    clf.fit(x_train.drop(['id'], axis=1), list(y_train))
    print(datetime.now().strftime('%Y-%m-%d %H:%M'))
    y_pred = clf.predict(x_train.drop(['id'], axis=1))
    # y_pred = clf.predict(x_train.drop(['id'], axis=1))
    y_predprob = clf.predict_proba(x_train.drop(['id'], axis=1))[:, 1]
    print("训练集自测Accuracy : %.4g" % accuracy_score(
        list(y_train), y_pred))  # Accuracy : 0.9852
    print("AUC Score (Train): %f" % roc_auc_score(list(y_train), y_predprob))
    
    print("模型训练完成")
    return clf


def draw_feature_score(clf, num=30):
    weight_kv = clf.get_booster().get_score()
    return nltk.FreqDist(weight_kv).plot(num)


def get_model_metrics(clf, x_train, rate_0=1, rate_1=1):
    print("---模型验证集评分---=")
    print(x_train.shape)
    X_test = c_data_sample.train_selected[~c_data_sample.train_selected.id.isin(
        list(x_train.id))]
    # X_test=c_data_sample.train
    test_1 = X_test[X_test.id.isin(list(label_1_df.id))]
    test_0 = X_test[X_test.id.isin(list(label_0_df.id))]

    X_test = pd.concat([test_0.sample(frac=rate_0),
                        test_1.sample(frac=rate_1)], ignore_index=True)
    print(X_test.shape)
    y_pred = clf.predict(X_test.drop(['id'], axis=1))
    print(y_pred[:30])
    y_predprob = clf.predict_proba(X_test.drop(['id'], axis=1))[:, 1]
    print(y_predprob[:30])
    y_test = pd.merge(X_test, labels, on='id').target
    print(list(y_test)[:30])
    print(dict(nltk.FreqDist(list(y_test))))
    print("Accuracy : %.4g" % accuracy_score(
        list(y_test), y_pred))  # Accuracy : 0.9852
    print("AUC Score (Train): %f" % roc_auc_score(list(y_test), y_predprob))

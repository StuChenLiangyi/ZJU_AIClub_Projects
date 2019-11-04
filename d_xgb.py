# coding=utf-8
import nltk
import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import b_feature_engineering
from b_feature_engineering import labels, label_0_df, label_1_df
from sklearn.model_selection import StratifiedKFold
import gc
import c_data_sample


def get_trained_xgb(x_train, scale_pos_weight):
    # 训练xgb模型
    print("开始训练模型")

    print(datetime.now().strftime('%Y-%m-%d %H:%M'))

    # Get folds for k-fold CV
    NFOLD = 3
    sfolds = StratifiedKFold(n_splits=NFOLD, random_state=0, shuffle=True)
    # folds = KFold(n_splits=NFOLD, shuffle=True, random_state=0)
    # fold = folds.split(df)
    x_train['target'] = b_feature_engineering.labels.target
#     x_train=x_train.head(1000).copy()
    # sfolds = KFold(n_splits=NFOLD, shuffle=True, random_state=0)
    sfolds.get_n_splits(x_train, x_train["target"])
    fold = sfolds.split(x_train, x_train["target"])

    label = 'target'
    predictors = list(x_train.columns.difference(
        ['id', 'target']))
    max_auc = 0

    for i, (train_index, test_index) in enumerate(fold):
        #         train_X, valid_X = x_train[predictors].values[train_index], x_train[predictors].values[test_index]
        #         train_y, valid_y = x_train[label].values[train_index], x_train[label].values[test_index]
        train_X, valid_X = x_train[predictors].loc[train_index], x_train[predictors].loc[test_index]
        train_y, valid_y = x_train[label].values[train_index], x_train[label].values[test_index]
        clf = xgb.XGBClassifier(max_depth=8, n_estimators=200,
                                scale_pos_weight=scale_pos_weight/8,
                                eval_metric='auc',
                                objective='binary:logistic',
                                # colsample_bytree=0.8,
                                # subsample=0.8,
                                nthread=32,
                                learning_rate=0.12
                                )

        clf.fit(train_X, train_y, early_stopping_rounds=50,
                                 eval_metric='auc', 
                eval_set=[(valid_X, valid_y)],
                verbose=50)
#         clf.fit(train_X, train_y)
        kv = clf.get_booster().get_score()
        print(list(kv.keys())[:30])
        print(datetime.now().strftime('%Y-%m-%d %H:%M'))
        y_pred = clf.predict(valid_X)
        # y_pred = clf.predict(x_train.drop(['id'], axis=1))
        y_predprob = clf.predict_proba(valid_X)[:, 1]

        print("训练集自测Accuracy : %.4g" % accuracy_score(
            list(valid_y), y_pred))  # Accuracy : 0.9852
        auc = clf.best_score
        evals_results = clf.evals_result()
#         print(evals_results)

        print("AUC Score (Train): %f" % auc)

    print("模型训练完成")
    if auc > max_auc:
        max_auc = auc
        print("auc>>>>>>", auc)
        clf_best = clf

    x_train = x_train.drop(['target'], axis=1)
    return clf_best


def draw_feature_score(clf, num=30):
    weight_kv = clf.get_booster().get_score()
    return nltk.FreqDist(weight_kv).plot(num)


def get_model_metrics(clf, x_train, rate_0=1, rate_1=1):
    print("---模型验证集评分---=")
    print(x_train.shape)
    columns = clf.get_booster().feature_names
#     print(columns)
    X_test = c_data_sample.train_selected[~c_data_sample.train_selected.id.isin(
        list(x_train.id))]

    # X_test=c_data_sample.train
    test_1 = X_test[X_test.id.isin(list(label_1_df.id))]
    test_0 = X_test[X_test.id.isin(list(label_0_df.id))]

    X_test = pd.concat([test_0.sample(frac=rate_0),
                        test_1.sample(frac=rate_1)], ignore_index=True)
    print(X_test.shape)
    X_test_temp = X_test[columns]
    y_pred = clf.predict(X_test_temp)
    print(y_pred[:30])
    y_predprob = clf.predict_proba(X_test_temp)[:, 1]
    print(y_predprob[:30])
    y_test = pd.merge(X_test, labels, on='id').target
    print(list(y_test)[:30])
    print(dict(nltk.FreqDist(list(y_test))))
    print("Accuracy : %.4g" % accuracy_score(
        list(y_test), y_pred))  # Accuracy : 0.9852
    print("AUC Score (Train): %f" % roc_auc_score(list(y_test), y_predprob))

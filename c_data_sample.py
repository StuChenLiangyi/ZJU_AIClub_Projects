# coding=utf-8
import numpy as np
import pandas as pd
import gc
import datetime
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
import os
from sklearn.externals import joblib
# import matplotlib.pyplot as plt
# import nltk

from imblearn.over_sampling import SMOTE
import b_feature_engineering

train_selected = None


def get_sampled_data(data, rate_0=0.7, rate_1=0.8):
    # global label_0_df, label_1_df
    # 采样，进行欠抽样和过抽样
    print("采样")
    train_1_df = data[data.id.isin(b_feature_engineering.label_1_df.id)]
    train_0_df = data[data.id.isin(b_feature_engineering.label_0_df.id)]
    train_0_sample = train_0_df.sample(frac=rate_0)
    train_1_sample = train_1_df.sample(frac=rate_1)
    # train_1_sample=pd.concat([train_1_sample,train_1_sample,train_1_sample,train_1_sample,train_1_sample],ignore_index=True)

    scale_pos_weight = 1.0*train_0_sample.shape[0]/train_1_sample.shape[0]
    # sum(negative cases) / sum(positive cases)
    print(scale_pos_weight)

    x_train = pd.concat([train_0_sample, train_1_sample], ignore_index=True)
    x_train = x_train.sample(frac=1)  # 乱序

    # X_train=x_train.copy()
    # Train=b_feature_engineering.train.copy()
    print(x_train.shape)
    del train_0_sample, train_1_sample
    gc.collect()
    return x_train, scale_pos_weight


def eliminate_different_data(neighbor_nums=130000):
    global train_selected
    print("neighbor_nums:", neighbor_nums)

    print("正在寻找测试集近邻点")
    train_0_df = b_feature_engineering.train.loc[b_feature_engineering.label_0_df.index].copy(
    )
    train_0_df['label'] = 1
    Test = b_feature_engineering.test.copy()
    Test['label'] = 0

    df = pd.concat([train_0_df, Test], ignore_index=True)
    del train_0_df, Test
    gc.collect()
    # Update column names
    predictors = list(df.columns.difference(
        ['id', 'label', 'disobey_num', 'disobey_rate', 'dist_disobey_rate', 'addr_disobey_rate',
            'card_disobey_rate', 'card_disobey_num',
            'certValidMonths', 'is_InValidStop',
            'missing_columns',
            'is_same_addr',
            'is_same_prov']))

    # Get label column name
    label = 'label'

    # lgb params
    lgb_params = {
        'boosting': 'gbdt',
        'application': 'binary',
        'metric': 'auc',
        'learning_rate': 0.2,
        'num_leaves': 32,
        'max_depth': 6,
        'bagging_fraction': 0.5,
        'bagging_freq': 5,
        'feature_fraction': 0.9,
        'is_unbalance': True
    }

    # Get folds for k-fold CV
    NFOLD = 10
    sfolds = StratifiedKFold(n_splits=NFOLD, random_state=0, shuffle=True)
    # folds = KFold(n_splits=NFOLD, shuffle=True, random_state=0)
    # fold = folds.split(df)

    # sfolds = KFold(n_splits=NFOLD, shuffle=True, random_state=0)
    sfolds.get_n_splits(df, df[label])
    fold = sfolds.split(df, df[label])

    eval_score = 0
    n_estimators = 0
    eval_preds = np.zeros(df.shape[0])

    # Run LightGBM for each fold
    for i, (train_index, test_index) in enumerate(fold):
        print("\n[{}] Fold {} of {}".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), i+1, NFOLD))
        train_X, valid_X = df[predictors].values[train_index], df[predictors].values[test_index]
        train_y, valid_y = df[label].values[train_index], df[label].values[test_index]
#             train_X=train_X.as_matrix()
#             valid_X=valid_X.as_matrix()
        dtrain = lgb.Dataset(train_X, label=train_y,
                                feature_name=list(predictors)
                                )
        dvalid = lgb.Dataset(valid_X, label=valid_y,
                                feature_name=list(predictors)
                                )

        eval_results = {}
        print("开始训练lightgbm")
        bst = lgb.train(lgb_params,
                        dtrain,
                        valid_sets=[dtrain, dvalid],
                        valid_names=['train', 'valid'],
                        evals_result=eval_results,
                        num_boost_round=500,
                        early_stopping_rounds=50,
                        verbose_eval=100)
        print("得到模型lightgbm")
        print("\nRounds:", bst.best_iteration)
        print("auc: ", eval_results['valid']['auc'][bst.best_iteration-1])

        n_estimators += bst.best_iteration
        eval_score += eval_results['valid']['auc'][bst.best_iteration-1]

        eval_preds[test_index] += bst.predict(valid_X,
                                                num_iteration=bst.best_iteration)

    n_estimators = int(round(n_estimators/NFOLD, 0))
    eval_score = round(eval_score/NFOLD, 6)

    print("\nModel Report")
    print("Rounds: ", n_estimators)
    print("AUC: ", eval_score)

    # Feature importance
    # print(lgb.plot_importance(bst, max_num_features=20))
    importance = bst.feature_importance(importance_type='split')
    feature_name = bst.feature_name()
    # for (feature_name,importance) in zip(feature_name,importance):
    #     print (feature_name,importance)
    feature_importance = pd.DataFrame(
        {'feature_name': feature_name, 'importance': importance})
    feature_importance = feature_importance.sort_values(
        by="importance", ascending=False)
    print(feature_importance.head(20))
    # Get training rows that are most similar to test
    df_av = df[['id', 'label']].copy()
    df_av['preds'] = eval_preds
    df_av_train = df_av[df_av.label == 1]
    df_av_train = df_av_train.sort_values(
        by=['preds']).reset_index(drop=True)
    # df_av_train.to_csv('df_similar_train.csv', index=0)
    # Check distribution

    # print(df_av_train.preds.hist())

    rate = df_av_train.shape[0]/df_av.shape[0]
    print(df_av_train[(df_av_train.preds >= 0.85)].preds.describe())

#         selected_id = list(df_av_train[df_av_train.preds <= rate].id)
    selected_id = list(df_av_train.head(neighbor_nums).id)  # 服了，直接选最接近的8w条
    selected_id += list(b_feature_engineering.label_1_df.id)
    train_selected = b_feature_engineering.train[b_feature_engineering.train.id.isin(
        selected_id)]
    print("已筛选出和测试集近邻点", len(selected_id))
    train_selected.to_csv('./results/train_selected.csv', index=0)
    return


def plot_resampling(ax, X, y, title):
    c0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5)
    c1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
#     ax.spines['left'].set_position(('outward', 10))
#     ax.spines['bottom'].set_position(('outward', 10))
#     ax.set_xlim([-6, 8])
#     ax.set_ylim([-6, 6])
    return c0, c1

# 采用smote算法人工生成新样本
# 输入训练集、训练集对应的标签、生成少样本的个数
# 返回包含新生成样本的新训练集,对应的标签


def make_new_sample(x_train, y_train, nums=1000):
    print(x_train.select_dtypes(exclude=['int64', 'float64']).columns)
    for col in x_train.columns:
        if x_train[col].isnull().any():
            print(col)
    print("------------------正在生成新样本-----------------------------")
    # Apply regular SMOTE
#     kind = ['regular', 'borderline1'] #两种样本生成方法
    kind = ['regular']
    sm = [SMOTE(kind=k, ratio={1: nums}) for k in kind]
    X_resampled = []
    y_resampled = []
#     X_res_vis = []
    for method in sm:
        X_res, y_res = method.fit_sample(x_train.drop(['id'], axis=1), y_train)
        X_resampled.append(X_res)
        y_resampled.append(y_res)
#         X_res_vis.append(pca.transform(X_res))
        print(method.kind)
    # Two subplots, unpack the axes array immediately
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # Remove axis for second plot
    ax2.axis('off')
    ax_res = [ax3, ax4]
    c0, c1 = plot_resampling(ax1, x_train.drop(
        ['id'], axis=1).values, y_train, 'Original set')
    ax2.legend((c0, c1), ('Class #0', 'Class #1'), loc='center',
               ncol=1, labelspacing=0.)

    for i in range(len(kind)):
        plot_resampling(ax_res[i], X_resampled[i], y_resampled[i],
                        'SMOTE {}'.format(kind[i]))
    f

    new = pd.DataFrame(X_resampled[0])  # 取regular方法产生的样本
    columns = x_train.columns.difference(
        ['id'])
    new.columns = x_train.drop(['id'], axis=1).columns

    new['id'] = range(new.shape[0])
    new['target'] = y_resampled[0]

    b_feature_engineering.train = new.drop(['target'], axis=1).copy()
    b_feature_engineering.labels = new[['id', 'target']].copy()
    b_feature_engineering.label_0_df = b_feature_engineering.labels[
        b_feature_engineering.labels.target == 0]
    b_feature_engineering.label_1_df = b_feature_engineering.labels[
        b_feature_engineering.labels.target == 1]
    del new
    gc.collect()
    print("=====新样本生成结束====", b_feature_engineering.train.shape)

import numpy as np
import pandas as pd
import gc
import datetime
import nltk

from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
import os
from sklearn.externals import joblib
# import matplotlib.pyplot as plt
# import nltk
import os
import b_feature_engineering
from b_feature_engineering import label_0_df, label_1_df

train_selected = None


def get_sampled_data(data, rate_0=0.7, rate_1=0.8):
    # global label_0_df, label_1_df
    # 采样，进行欠抽样和过抽样
    print("采样")
    #最接近的500条留着做验证集
    train_1_df = (data.tail(80500))[data.id.isin(label_1_df.id)]
    train_0_df = (data.tail(80500))[data.id.isin(label_0_df.id)]
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


def eliminate_different_data():
    global train_selected
    if(os.path.exists('./results/train_selected.csv')):
        train_selected = pd.read_csv('./results/train_selected.csv')
        train_selected.index = train_selected.id
        print(train_selected.shape)
        print("检测到已完成测试集近邻点收集")
    else:
        print("正在寻找测试集近邻点")
        train_0_df = b_feature_engineering.train.loc[label_0_df.index].copy()
        train_0_df['label'] = 1
        Test = b_feature_engineering.test.copy()
        Test['label'] = 0

        df = pd.concat([train_0_df, Test], ignore_index=True)
        del train_0_df, Test
        gc.collect()
        # Update column names
        predictors = list(df.columns.difference(
            ['id', 'label', 'disobey_num', 'disobey_rate','dist_disobey_rate','addr_disobey_rate', 
            'card_disobey_rate', 'card_disobey_num']))


        # Get label column name
        label = 'label'

        # lgb params
        lgb_params = {
            'boosting': 'gbdt',
            'application': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'num_leaves': 32,
            'max_depth': 6,
            'bagging_fraction': 0.5,
            'bagging_freq': 5,
            'feature_fraction': 0.9,
            'is_unbalance':True
        }

        # Get folds for k-fold CV
        NFOLD = 5
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
                            early_stopping_rounds=100,
                            verbose_eval=100)
            print("得到模型lightgbm")
            print("\nRounds:", bst.best_iteration)
            print("AUC: ", eval_results['valid']['auc'][bst.best_iteration-1])

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
        feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} )
        feature_importance=feature_importance.sort_values(by="importance" , ascending=False)
        print(feature_importance.head(20) )
        # Get training rows that are most similar to test
        df_av = df[['id', 'label']].copy()
        df_av['preds'] = eval_preds
        df_av_train = df_av[df_av.label == 1]
        df_av_train = df_av_train.sort_values(
            by=['preds']).reset_index(drop=True)
        # df_av_train.to_csv('df_similar_train.csv', index=0)
        # Check distribution

        # print(df_av_train.preds.hist())
        
        # rate = df_av_train.shape[0]/df_av.shape[0]
        print(df_av_train[(df_av_train.preds>=0.9) & (df_av_train.preds<= 0.99)].preds.describe())

        # selected_id = list(df_av_train[df_av_train.preds <= 0.93].id)
        selected_id = list(df_av_train.head(80000).id)#服了，直接选最接近的8w条
        selected_id += list(label_1_df.id)
        train_selected = b_feature_engineering.train[b_feature_engineering.train.id.isin(
            selected_id)]
        print("已筛选出和测试集近邻点", len(selected_id))
        train_selected.to_csv('./results/train_selected.csv', index=0)
        return
        

import os
from sklearn.externals import joblib
import c_data_sample
from d_xgb import get_trained_xgb, get_model_metrics
import gc
from itertools import combinations
import pandas as pd
import b_feature_engineering
from b_feature_engineering import labels


def get_mul_features(top_features):
    # 得到top7依次组合下的top15*7新乘法特征,每次均采用不同的样本
    if(os.path.exists('./results/mul_features_top.kv')):
        mul_features_top = joblib.load('./results/mul_features_top.kv')
    else:
        mul_features_top = {}
    for x in top_features:
        if x not in list(mul_features_top.keys()):
            x_train, scale_pos_weight = c_data_sample.get_sampled_data(c_data_sample.train_selected,
                                                                       0.6, 0.9)
            for col in x_train.columns:
                if col not in top_features and col != 'id':
                    x_train[col] = x_train[x] * x_train[col]
                    x_train.rename(
                        columns={col: (x+'_mul_'+col)}, inplace=True)
            x_train = x_train.drop(top_features, axis=1)
            clf = get_trained_xgb(x_train, scale_pos_weight)
            weight_kv = clf.get_booster().get_score()
            mul_features_top[x] = list(weight_kv.keys())[:15]
            del x_train
            gc.collect()
            joblib.dump(mul_features_top, './results/mul_features_top.kv')
        print(mul_features_top[x])

    # top内部组合
    if ('top_mul_internal') not in list(mul_features_top.keys()):
        print("mul 内部组合")
        x_train, scale_pos_weight = c_data_sample.get_sampled_data(c_data_sample.train_selected,
                                                                   0.6, 0.9)
        df = pd.DataFrame()
        df['id'] = x_train.id
        # print("特征维度",len(list(combinations(top_features,2))))
        for (x, y) in list(combinations(top_features, 2)):
            df[x+'_mul_'+y] = x_train[x] * x_train[y]
        print(df.shape)
        clf = get_trained_xgb(df, scale_pos_weight)
        weight_kv = clf.get_booster().get_score()
        mul_features_top['top_mul_internal'] = list(weight_kv.keys())[:15]
        del x_train
        gc.collect()
        joblib.dump(mul_features_top, './results/mul_features_top.kv')
    # else:
    #     mul_features_top.pop('top_mul_internal')
    #     joblib.dump(mul_features_top, './results/mul_features_top.kv')
    #     print("已删")


def get_add_features(top_features):
    # 得到top7依次组合下的top15*7新加法特征,每次均采用不同的样本
    if(os.path.exists('./results/add_features_top.kv')):
        add_features_top = joblib.load('./results/add_features_top.kv')
    else:
        add_features_top = {}
    for x in top_features:
        if x not in list(add_features_top.keys()):
            x_train, scale_pos_weight = c_data_sample.get_sampled_data(c_data_sample.train_selected,
                                                                       0.6, 0.9)
            for col in x_train.columns:
                if col not in top_features and col != 'id':
                    x_train[col] = x_train[x] + x_train[col]
                    x_train.rename(
                        columns={col: (x+'_add_'+col)}, inplace=True)
            x_train = x_train.drop(top_features, axis=1)
            clf = get_trained_xgb(x_train, scale_pos_weight)
            weight_kv = clf.get_booster().get_score()
            add_features_top[x] = list(weight_kv.keys())[:15]
            del x_train
            gc.collect()
            joblib.dump(add_features_top, './results/add_features_top.kv')
        print(add_features_top[x])

    # top内部组合
    if ('top_add_internal') not in list(add_features_top.keys()):
        print("add 内部组合")
        x_train, scale_pos_weight = c_data_sample.get_sampled_data(c_data_sample.train_selected,
                                                                   0.6, 0.9)
        df = pd.DataFrame()
        df['id'] = x_train.id
        # print("特征维度",len(list(combinations(top_features,2))))

        for (x, y) in list(combinations(top_features, 2)):
            df[x+'_add_'+y] = x_train[x] + x_train[y]
        print(df.shape)
        clf = get_trained_xgb(df, scale_pos_weight)
        weight_kv = clf.get_booster().get_score()
        add_features_top['top_add_internal'] = list(weight_kv.keys())[:15]
        del x_train
        gc.collect()
        joblib.dump(add_features_top, './results/add_features_top.kv')
    # else:
    #     add_features_top.pop('top_add_internal')
    #     joblib.dump(add_features_top, './results/add_features_top.kv')
    #     print("已删")


def apply_combine_features():
    print("正在应用组合特征")
    add_features_top = joblib.load('./results/add_features_top.kv')
    mul_features_top = joblib.load('./results/mul_features_top.kv')

    for v in add_features_top.values():
        for col in v:
            cols = str(col).split('_add_')
            c_data_sample.train_selected[col] = c_data_sample.train_selected[cols[0]
                                                                             ]+c_data_sample.train_selected[cols[1]]
            b_feature_engineering.test[col] = b_feature_engineering.test[cols[0]
                                                                         ]+b_feature_engineering.test[cols[1]]

    for v in mul_features_top.values():
        for col in v:
            cols = str(col).split('_mul_')
            c_data_sample.train_selected[col] = c_data_sample.train_selected[cols[0]
                                                                             ]*c_data_sample.train_selected[cols[1]]
            b_feature_engineering.test[col] = b_feature_engineering.test[cols[0]
                                                                         ]*b_feature_engineering.test[cols[1]]
    print("组合特征应用完毕")
    print("特征排序：")
    if not os.path.exists('./results/clf_1.model'):
        # x_train, scale_pos_weight = c_data_sample.get_sampled_data(c_data_sample.train_selected,0.95, 0.9)
        print("全部训练")
        # clf = get_trained_xgb(x_train, scale_pos_weight)
        c_data_sample.train_selected.reset_index(
            drop=True, inplace=True)
        clf = get_trained_xgb(c_data_sample.train_selected, 80)
        # joblib.dump(clf,'./results/clf_1.model')
        weight_kv = clf.get_booster().get_score()
        # get_model_metrics(clf,x_train)
        print(list(weight_kv.keys())[:30])

        # X=c_data_sample.train_selected[~c_data_sample.train_selected.id.isin(
        #     list(x_train.id))]

        # y=pd.merge(X, labels, on='id').target
        # clf.fit(X.drop(['id'],axis=1),list(y),xgb_model =clf)
        joblib.dump(clf, './results/clf_1.model')
        # weight_kv = clf.get_booster().get_score()
        # print(list(weight_kv.keys())[:30])
        # joblib.dump(clf,'./results/feature_rank.kv')
        print("特征排序结束")
        y_predprob = clf.predict_proba(
            b_feature_engineering.test.drop(['id'], axis=1))[:, 1]
        df = pd.DataFrame()
        df['id'] = b_feature_engineering.test.id
        df['target'] = y_predprob
        df.to_csv('./results/submission_2.csv', index=0, header=True)
        # del x_train
        # gc.collect()
    else:
        print("检测到已完成排序")
        # if not os.path.exists('./results/submission_1.csv'):
        #     clf_1 = joblib.load('./results/clf_1.model')
        #     y_predprob = clf_1.predict_proba(b_feature_engineering.test.drop(['id'], axis=1))[:, 1]
        #     df=pd.DataFrame()
        #     df['id']=b_feature_engineering.test.id
        #     df['target']=y_predprob
        #     df.to_csv('./results/submission_1.csv',index=0,header=True)


def select_features():
    clf_1 = joblib.load('./results/clf_1.model')

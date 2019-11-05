import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as ctb
import nltk
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


def output(model, test, filename='./results/submission_0.csv'):

    try:
        columns = model.get_booster().feature_names
        test = test[['id']+columns]
    except:
        pass
    y_predprob = model.predict_proba(
        test.drop(['id'], axis=1))[:, 1]
    df = pd.DataFrame()
    df['id'] = test.id
    df['target'] = y_predprob
    df.to_csv(filename, index=0, header=True)
    df['random_ans'] = 0
    df.random_ans[df.id.isin(df.id.sample(frac=0.002))] = 1
    print("预测结果已经保存在", filename)
    print(df.target.describe())
    num_1 = df[df.target > 0.5].shape[0]
    print("-------------------------该预测结果中预测为1样本的数目为", num_1)
    print("-------------------------预测线上得分为", roc_auc_score(df.random_ans, y_predprob))
    if (num_1 < 10 or num_1 >200) :
        print("-----------------------------该结果可能不适合提交",filename)

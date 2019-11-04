import pandas as pd
import xgboost as xgb
import catboost as ctb

def output(model,test,filename='./results/submission_0.csv'):
            y_predprob = model.predict_proba(
                test.drop(['id'], axis=1))[:, 1]
            df=pd.DataFrame()
            df['id']=test.id
            df['target']=y_predprob
            df.to_csv(filename,index=0,header=True)
            print("预测结果已经保存在",filename)
            print(df.target.describe())

import catboost as ctb
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold

def get_trained_ctb(x_train, labels, scale_pos_weight=130):

    print("开始训练模型")

    print(datetime.now().strftime('%Y-%m-%d %H:%M'))
    x_train['target'] = labels
    # Get folds for k-fold CV
    NFOLD = 5
#     sfolds = StratifiedKFold(n_splits=NFOLD, random_state=0, shuffle=True)
    folds = KFold(n_splits=NFOLD, shuffle=True, random_state=0)
    fold = folds.split(x_train)
    
#     x_train=x_train.head(1000).copy()
    # sfolds = KFold(n_splits=NFOLD, shuffle=True, random_state=0)
#     sfolds.get_n_splits(x_train, x_train["target"])
#     fold = sfolds.split(x_train, x_train["target"])

    label = 'target'
    predictors = list(x_train.columns.difference(
        ['id', 'target']))
    max_auc = 0

    for i, (train_index, test_index) in enumerate(fold):
        #         train_X, valid_X = x_train[predictors].values[train_index], x_train[predictors].values[test_index]
        #         train_y, valid_y = x_train[label].values[train_index], x_train[label].values[test_index]
        train_X, valid_X = x_train[predictors].loc[train_index], x_train[predictors].loc[test_index]
        train_y, valid_y = x_train[label].values[train_index], x_train[label].values[test_index]
        clf = ctb.CatBoostClassifier(iterations=500, depth=6,
                                   #                                    cat_features=categorical_features_indices,
                                   learning_rate=0.1, loss_function='Logloss',                                                  logging_level='Verbose',
                                   custom_metric='AUC',
                                   eval_metric='AUC', scale_pos_weight=scale_pos_weight)

        clf.fit(train_X, train_y, early_stopping_rounds=50,
              eval_set=(valid_X, valid_y), verbose_eval=50,
              plot=True)
#         clf.fit(train_X, train_y)
        
        print(datetime.now().strftime('%Y-%m-%d %H:%M'))
        y_pred = clf.predict(valid_X)
        # y_pred = clf.predict(x_train.drop(['id'], axis=1))
        y_predprob = clf.predict_proba(valid_X)[:, 1]

        print("catboost 训练集自测Accuracy : %.4g" % accuracy_score(
            list(valid_y), y_pred))  # Accuracy : 0.9852
        auc = clf.best_score_['validation']['AUC']  # evals_results = clf.evals_result()
        # print(auc)

#         print(evals_results)

        print("AUC Score (Train): %f" % auc)

        print("模型训练完成")
        if auc > max_auc:
            max_auc = auc
            print("max auc>>>>>>", auc)
            clf_best = clf

    x_train = x_train.drop(['target'], axis=1)
    print("返回得分最高的模型",clf_best.best_score_['validation']['AUC'])
    return clf_best

    # x_train = x_train.drop(['id'], axis=1)
    # X_train, X_validation, y_train, y_validation = train_test_split(x_train, labels,
    #                                                                 test_size=0.3,
    #                                                                 random_state=1234)

#     categorical_features_indices = np.where(X_train.dtypes == np.float)[0]
    # model = ctb.CatBoostClassifier(iterations=500, depth=8,
    #                                #                                    cat_features=categorical_features_indices,
    #                                learning_rate=0.1, loss_function='Logloss',                                                  logging_level='Verbose',
    #                                custom_metric='AUC',
    #                                eval_metric='AUC', scale_pos_weight=scale_pos_weight)  


    # model.fit(X_train, y_train, early_stopping_rounds=50,
    #           eval_set=(X_validation, y_validation), verbose_eval=50,
    #           plot=True)
    # return model

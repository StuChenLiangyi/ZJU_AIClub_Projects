import catboost as ctb
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold


def get_trained_ctb(x_train, labels, scale_pos_weight=130):

    print("开始训练模型")

    print(datetime.now().strftime('%Y-%m-%d %H:%M'))
    x_train['target'] = labels
    x_train.index = x_train.id.values
    # Get folds for k-fold CV
    NFOLD = 2
#     sfolds = StratifiedKFold(n_splits=NFOLD, random_state=0, shuffle=True)
#     folds = KFold(n_splits=NFOLD, shuffle=True, random_state=0)
#     fold = folds.split(x_train)

#     x_train=x_train.head(1000).copy()
    sfolds = KFold(n_splits=NFOLD, shuffle=True, random_state=0)
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
        clf = ctb.CatBoostClassifier(iterations=500, depth=8,

                                     learning_rate=0.1, loss_function='Logloss',                                                  logging_level='Verbose',
                                     custom_metric='AUC',
                                     eval_metric='AUC', scale_pos_weight=scale_pos_weight)

        clf.fit(train_X, train_y, early_stopping_rounds=100,
                eval_set=(valid_X, valid_y), verbose_eval=100, use_best_model=True,
                plot=True)

        clf_xgb = xgb.XGBClassifier(max_depth=4, n_estimators=200,
                                    scale_pos_weight=scale_pos_weight/2,
                                    eval_metric='auc',
                                    objective='binary:logistic',
                                    eta=1,
                                    colsample_bytree=0.5,
                                    gamma=0,
                                    reg_lambda=1,
                                    reg_alpha=1,

                                    subsample=0.8,
                                    min_child_weight=100,

                                    learning_rate=0.1
                                    )
        print(i, "----------------开始训练xgb模型-------------------")
        clf_xgb.fit(train_X, train_y, early_stopping_rounds=100,
                    eval_metric='auc',
                    eval_set=[(valid_X, valid_y)],
                    verbose=100)
#         clf.fit(train_X, train_y)

        print(datetime.now().strftime('%Y-%m-%d %H:%M'))
        y_pred = clf.predict(valid_X)
        # y_pred = clf.predict(x_train.drop(['id'], axis=1))
        y_predprob = clf.predict_proba(valid_X)[:, 1]
        print("catboost AUC Score (Train): %f" %
              roc_auc_score(valid_y, y_predprob))
        print("catboost Accuracy : %.4g" % accuracy_score(
            list(valid_y), y_pred))  # Accuracy : 0.9852
        # evals_results = clf.evals_result()
        auc = clf.best_score_['validation']['AUC']
        # print(auc)

#         print(evals_results)

#         kv = clf.get_booster().get_score()
#         print(list(kv.keys())[:30])
#         print(datetime.now().strftime('%Y-%m-%d %H:%M'))
        y_pred = clf_xgb.predict(valid_X, ntree_limit=clf_xgb.best_iteration)
#         y_pred = clf.predict(x_train.drop(['id'], axis=1))
        y_predprob = clf_xgb.predict_proba(
            valid_X, ntree_limit=clf_xgb.best_iteration)[:, 1]

        print("xgb Accuracy : %.4g" % accuracy_score(
            list(valid_y), y_pred))  # Accuracy : 0.9852

        print("xgb AUC Score (Train): %f" % roc_auc_score(valid_y, y_predprob))

#         print(evals_results)

        print("模型训练完成")
        if auc > max_auc:
            max_auc = auc
            print("max auc>>>>>>", auc)
            clf_best = clf

    x_train = x_train.drop(['target'], axis=1)
    print("返回得分最高的catboost模型", clf_best.best_score_['validation']['AUC'])
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

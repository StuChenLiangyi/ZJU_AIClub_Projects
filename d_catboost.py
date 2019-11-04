import catboost as ctb
from sklearn.model_selection import train_test_split


def get_trained_ctb(x_train, labels, scale_pos_weight=130):
    x_train = x_train.drop(['id'], axis=1)
    X_train, X_validation, y_train, y_validation = train_test_split(x_train, labels,
                                                                    test_size=0.2,
                                                                    random_state=1234)

#     categorical_features_indices = np.where(X_train.dtypes == np.float)[0]
    model = ctb.CatBoostClassifier(iterations=500, depth=8,
                                   #                                    cat_features=categorical_features_indices,
                                   learning_rate=0.1, loss_function='Logloss',                                                  logging_level='Verbose',
                                   custom_metric='AUC',
                                   eval_metric='AUC', scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train, early_stopping_rounds=50,
              eval_set=(X_validation, y_validation), plot=True)
    return model

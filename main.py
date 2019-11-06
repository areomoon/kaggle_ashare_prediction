import numpy as np
import pandas as pd
from dataloader import dataloader
# from dataproc import dataproc
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt

def main(mode='process'):
    if mode == 'process':
        train_df = pd.read_csv('../train.csv')
        test_df = pd.read_csv('../test.csv')
        weather_tr = pd.read_csv('../weather_train.csv')
        weather_ts = pd.read_csv('../weather_test.csv')
        building_df = pd.read_csv('../building_metadata.csv')

        X, Y = dataloader(train_df=train_df, weather_df= weather_tr, building_df=building_df , mode='train').proc_pipeline()
        x_test = dataloader(train_df=test_df, weather_df= weather_ts, building_df=building_df, mode='test').proc_pipeline()
        # X, x_test = dataproc(train=X, test=x_test).proc_pipeline()

        X.to_csv('x_train.csv', index=False)
        Y.to_csv('y_train.csv', index=False)
        x_test.to_csv('x_test.csv', index=False)

    if not mode == 'process':
        X = pd.read_csv('X.csv')
        Y = pd.read_csv('Y.csv')
        x_test = pd.read_csv('x_test.csv')


    param = {
        "num_leaves": 40,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "objective": "regression",
        "boosting": "gbdt",
        "metric": "rmse",
        "verbose":1,
        'seed':7}

    categorical_features = ["building_id", "site_id", "meter", "primary_use", "hour", "weekday",'month']
    target_col = ['target1']


    nfold = 5
    kf = KFold(n_splits=nfold, random_state=227, shuffle=True)
    for cols in target_col:
        print('Training and predicting for target {}'.format(cols))
        all_lb = np.zeros(len(X))
        oof = np.zeros(len(X))
        all_pred = np.zeros(len(x_test))
        n = 1
        for train_index, valid_index in kf.split(Y[cols]):
            print("fold {}".format(n))
            lgb_train = lgb.Dataset(X.iloc[train_index],
                                    label=Y[cols].iloc[train_index].values,
                                    categorical_feature=categorical_features,
                                    free_raw_data=False
                                    )
            lgb_valid = lgb.Dataset(X.iloc[valid_index],
                                    label=Y[cols].iloc[train_index].values,
                                    categorical_feature=categorical_features,
                                    free_raw_data=False
                                    )
            lgb_model = lgb.train(param,
                                  lgb_train,
                                  15000,
                                  valid_sets=[lgb_train, lgb_valid],
                                  verbose_eval=200,
                                  early_stopping_rounds=200)

            all_lb[valid_index] = Y[cols].iloc[valid_index].values
            oof[valid_index] = lgb_model.predict(x_test.iloc[valid_index], num_iteration=lgb_model.best_iteration)
            all_pred += lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration) / nfold
            n = n + 1
        print("\n\nCV RMSE: {:<0.4f}".format(np.sqrt(mean_squared_error(all_lb, oof))))

    #     all_lb = all_lb.reshape(1, -1)
    #     oof = oof.reshape(1, -1)
    #     all_pred = all_pred.reshape(1, -1)
    #     all_lbs.append(all_lb)
    #     all_oof.append(oof)
    #     all_preds.append(all_pred)
    #
    # final_preds = np.hstack(all_preds).reshape(-1)
    # final_preds[final_preds < 0] = 0
    pred = np.expm1(np.clip(all_pred, 0, a_max=None))

    # save
    sub = pd.read_csv('../sample_submission.csv')
    sub["meter_reading"] = pred
    sub.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    main(mode='processed')
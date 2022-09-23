import os
import sys
from copy import deepcopy
import multiprocessing

import pickle
import numpy as np
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


np.random.seed(0)


ROOT_DIR = os.environ.get('ROOT_DIR')
SKEMPI2_DIR = os.environ.get('SKEMPI2_DIR')


def main():
    # outdir = f'{ROOT_DIR}/models/GeoPPI/trainedmodels/SKEMPI2-ML/xgb-big'
    outdir = f'{ROOT_DIR}/models/GeoPPI/trainedmodels/SKEMPI2-ML/rf'
    os.makedirs(outdir)
    modeloutfile = f'{outdir}/cv_model.pkl'
    logfile = f'{outdir}/log.txt'
    loghandle = open(logfile, 'w')
    sys.stdout = loghandle
    sys.errout = sys.stdout

    # Read data
    data_path = f'{SKEMPI2_DIR}/SKEMPI2-ML-trainval-GeoPPI-X-y.pkl'
    folds_path = f'{SKEMPI2_DIR}/SKEMPI2-ML-trainval-CV-5-FoldX.pkl'
    with open(data_path, 'rb') as file:
        X, y = pickle.load(file)
    with open(folds_path, 'rb') as file:
        cv = pickle.load(file)

    # Init model
    # XGBoost regressor
    # regressor = xgb.XGBRegressor(
    #     n_jobs=multiprocessing.cpu_count() // 2,
    #     seed=0
    # )
    # Random Forest regressor
    regressor = RandomForestRegressor(
        n_jobs=multiprocessing.cpu_count() // 2
    )
    regressor_baseline = deepcopy(regressor)

    # Calculate baseline scores
    untrained_scores_normal = []
    untrained_scores_shuffle = []
    for train_idx, val_idx in cv:
        # Normal distribution MLE
        mean, std = y[train_idx].mean(), y[train_idx].std()
        y_pred = np.random.normal(mean, std, y[val_idx].shape)
        score = -sklearn.metrics.mean_squared_error(y[val_idx], y_pred)
        untrained_scores_normal.append(score)

        # CV score on shuffled labels
        X_shuffled = X[train_idx]
        y_shuffled = np.random.permutation(y[train_idx])
        regressor_baseline.fit(X_shuffled, y_shuffled)
        y_pred = regressor_baseline.predict(X[val_idx])
        score = -sklearn.metrics.mean_squared_error(y[val_idx], y_pred)
        untrained_scores_shuffle.append(score)

    print('Untrained scores normal:', untrained_scores_normal)
    print('Mean untrained score normal:', np.mean(untrained_scores_normal))
    print('Untrained scores shuffle:', untrained_scores_shuffle)
    print('Mean untrained score shuffle:', np.mean(untrained_scores_shuffle))


    # Optimize hyper-parameters
    # "Small" XGBoost
    # param_grid = {
    #     'n_estimators': [10, 50, 100, 200],
    #     'max_depth': [2, 4, 6],
    #     'min_child_weight': [1, 3, 5],
    #     'gamma': [i/10 for i in [0, 2, 4]],
    #     'reg_alpha': [1e-5, 0.1, 1, 100],
    #     'learning_rate': [0.1, 0.01]
    # }
    # "Big" XGBoost
    # param_grid = {
    #     'n_estimators': [200, 300, 400, 500, 600],
    #     'max_depth': [6],
    #     'min_child_weight': [5],
    #     'gamma': [0.2],
    #     'reg_alpha': [100],
    #     'learning_rate': [0.1, 0.01]
    # }
    # Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300,  500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 3, 4, 6, 8]
    }
    model = GridSearchCV(
        regressor,
        param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=51,
        n_jobs=2,
        refit=False
    )

    # Fit CV
    model.fit(X, y)

    # Print stats
    print(model.best_score_)
    print(model.best_params_)
    print(model.cv_results_)

    # Save
    with open(modeloutfile, 'wb') as file:
        pickle.dump(model, file)

    loghandle.close()


if __name__ == '__main__':
    main()

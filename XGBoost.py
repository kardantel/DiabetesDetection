import warnings
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
from utils import Utils

utils = Utils()

sns.set_style("white")
warnings.filterwarnings('ignore')


class XGB:
    def __init__(self, df, nameDS, rs=None, prePros=False, model=None, solver=None):
        '''
        Returns the results of using the XGBoost technique
        with PCA, LDA, and no-preprocessing techniques.

        ## Parameters
        df: array-like of shape (n_samples, n_features)
            Dataset to work with. Can be for example a list, or an array.
        nameDS: str. Dataset name to work with.
        rs: int, default=1
            `random_state` value from `train_test_split` method. it's configured
            by default with `1` so that the user uses the one they want.
        prePros: boolean, default=False
            Allows you to preprocess the data in the `df` variable.
            If it is False (by default) it avoids preprocessing.
        model: object, default=None
            Receives the best model obtained when the data is not preprocessed
            in the variable `df`.
        solver: 'pca', 'ipca', 'lda', default=None
            Receive the type of solver to perform data preprocessing.
        '''
        self.df = df
        #self.nameDS = nameDS
        self.best_score = 9999
        self.best_model = None

        print(f'Ejecutando XGBoost para {nameDS}...')
        self.reg = {'XGBC': XGBClassifier(verbosity=0)}
        self.params = {'XGBC': {'n_estimators': [50, 100, 150, 200, 250, 300, 350,
                                                 400, 450, 500, 550, 600, 650, 700,
                                                 750, 800, 850, 900, 950, 1000],
                                'learning_rate': [0.001, 0.01, 0.05, 0.1],
                                'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                                'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                'colsample_bytree': [.1, .2, .3, .4, .5, .6, .7, .8, .9],
                                'reg_lambda': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                                               0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
                                               0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
                                               0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,
                                               0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
                                               0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
                                               0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
                                               0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
                                               0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
                                               0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1],
                                'subsample': [.1, .2, .3, .4, .5, .6, .7, .8, .9]}
                       }

        if prePros:
            # For this part to work it is completely necessary to have executed the code without preprocessing and to use the following hyperparameters: prePros=True, model=xgbc_xX.best_model, solver='pca' or 'lda'.
            self.X_std, self.y = utils.features_target(
                self.df, ['has_DM2'], ss=True)
            self.X_train_std, self.X_test_std, self.y_train_std, self.y_test_std = train_test_split(self.X_std, self.y,
                                                                                                    test_size=0.35,
                                                                                                    random_state=rs)

            # Dimensionality reduction techniques are instantiated and trained with the best hyperparameters.
            self.y_pred_pca = utils.dimReduction(model, self.X_train_std, self.X_test_std,
                                                 self.y_train_std, self.y_test_std, solver=solver)

            print(f'Metrics for {nameDS} with preprocessing...')

            utils.printROC(self.y_test_std, self.y_pred_pca)

            self.classes = np.unique(self.y)

            print(classification_report(self.y_test_std,
                                        self.y_pred_pca, labels=self.classes))

            self.conMatrix_pca = confusion_matrix(
                self.y_test_std, self.y_pred_pca)

            utils.printMatrix(self.y_test_std, self.conMatrix_pca)

        else:
            self.X, self.y = utils.features_target(self.df, ['has_DM2'])
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                    test_size=0.35,
                                                                                    random_state=1)

            self.classes = np.unique(self.y)

            print(
                f'   Looking for the best model for {nameDS} without preprocessing..')
            # Loops through each item in the 'reg' dictionary.
            for name, reg in self.reg.items():
                grid_reg = RandomizedSearchCV(reg,               # Regulators.
                                              # Parameter dictionaries.
                                              self.params[name],
                                              cv=10).fit(self.X, self.y)
                score = np.abs(grid_reg.best_score_)

                if score < self.best_score:
                    self.best_score = score
                    self.best_model = grid_reg.best_estimator_

            utils.model_export(nameDS, self.best_model, self.best_score)

            print(f'Metrics for {nameDS} with  XGBoost...')

            self.XGB_nP = self.best_model.fit(self.X_train, self.y_train)

            self.y_pred = self.XGB_nP.predict(self.X_test)

            utils.printROC(self.y_test, self.y_pred)

            print(classification_report(self.y_test,
                                        self.y_pred, labels=self.classes))

            self.conMatrix = confusion_matrix(self.y_test, self.y_pred)

            utils.printMatrix(self.y_test, self.conMatrix)

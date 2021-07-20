'''
File that contains all the methods used by the classifiers.
'''

import warnings
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
warnings.filterwarnings('ignore')


class Utils:
    '''
    Class that contains all the methods used by the classifiers.
    '''
    @classmethod
    def load_from_csv(cls, path, clear=False):
        '''
        Returns a list with the datasets from a specified address.

        ## Parameters
        path: address where the datasets are stored.
        clear: boolean [optional], default=False
            If it is True, the column `patientID` and row 0 are eliminated:
            `ramanShift`, leaving the dataset with only numeric variables.
            The default value `clear=False`, prevents cleaning.
        '''
        csvlist = os.listdir(path)
        raman_csv = []

        if clear:
            for raman in csvlist:
                csv = pd.read_csv(f'{path}/{raman}')

                obj = csv.dtypes == object
                obj_cols = [c for c in obj.index if obj[c]]

                # Deletes the categorical column.
                csvNum = csv.drop(obj_cols, axis=1)
                csvNum = csvNum.drop(0)  # Deletes the first row.
                raman_csv.append(csvNum)
            return raman_csv
        else:
            for raman in csvlist:
                csv = pd.read_csv(f'{path}/{raman}')
                raman_csv.append(csv)
            return raman_csv

    @classmethod
    def features_target(cls, dataset, y, ss=False):
        '''
        Returns the data separated in features and objective.

        ## Parameters
        dataset: array-like of shape (n_samples, n_features)
            Original dataset to work with. Can be for example a list, or an
            array.
        y: str
            Name of the target column. In this specific project it is
            `['has_DM2']`.
        ss: boolean [optional], default=False
            If it is `True`, center the data before scaling to zero mean and
            scale the data to unit variance.
            The default value `ss=False`, prevents standar scaler.
        '''
        if ss:
            X = dataset.drop(y, axis=1)
            X_std = StandardScaler().fit_transform(X)
            X_std = pd.DataFrame(X_std, columns=X.columns)
            y = dataset[y]
            return X_std, y
        else:
            X = dataset.drop(y, axis=1)
            y = dataset[y]
            return X, y

    @classmethod
    def model_export(cls, nameDS, clf, score):
        '''
        Prints the best model found and its final score.

        ## Parameters
        nameDS: str. Dataset name to work with.
        clf: object
            Best model obtained from hyperparameter search.
        score: float
            Best score obtained from hyperparameter search.
        '''
        print(f'   From {nameDS}:', 'Best model:',
              clf, '| Best score:', score)

    @classmethod
    def printMatrix(cls, y_true, conMatrix):
        '''
        Prints the confusion matrix.

        ## Parameters
        y_true: array-like of shape (n_samples, n_features)
            `y_test` variable obtained from separation with `train_test_split`.
        conMatrix: ndarray of shape (n_classes, n_classes)
            Confusion matrix obtained from `confusion_matrix()`.
        '''
        # Defines the values for each group.
        class_names = np.unique(y_true)
        fig, ax = plt.subplots()
        # Creates a vector with the amount of values in 'class_names' from 0.
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        sns.heatmap(pd.DataFrame(conMatrix),
                    annot=True, cmap="Blues_r", fmt="g")
        plt.tight_layout()
        plt.title("Confusion matrix")
        plt.ylabel("Current Label")
        plt.xlabel("Prediction Label")

    @classmethod
    def printROC(cls, y_test, y_pred):
        '''
        Prints the ROC curve.

        ## Parameters
        y_test: array-like of shape (n_samples,)
            Variable obtained from separation with `train_test_spli`.
        y_pred: array-like of shape (n_samples,)
            Variable obtained from using the `predict()` function.
        '''
        fpr, tpr, _ = roc_curve(y_test, y_pred)

        plt.plot(fpr, tpr, color='darkorange', lw=2, marker='.',
                 label='ROC curve (area = %0.4f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    @classmethod
    def dimReduction(cls, model, X_train, X_test, y_train,
                     y_test, solver='pca'):
        '''
        It uses the PCA, iPCA, and LDA dimensionality reduction methods.
        Returns the accuracy value and the prediction value according to method

        ## Parameters
        model: object
            Model that will be used to carry out the dimensionality reduction.
        X_train: array-like of shape (n_samples, n_features)
            Variable for training.
        X_test: array-like of shape (n_samples, n_features)
            Variable for tests.
        y_train: array-like of shape (n_samples,)
            Target variable for training.
        y_test: array-like of shape (n_samples,)
            Target variable for tests.
        solver: str, 'pca', 'ipca', 'lda' default='pca'
            Variable that indicates which dimensionality reduction technique
            to choose.
        '''
        if solver == 'pca':
            pca_auto = PCA()
            model_pca_auto = pca_auto.fit(X_train)

            print("  Calculating the best component and best score for PCA...")
            best_score = 0
            best_numbr = 0

            for n in np.arange(model_pca_auto.n_components_) + 1:
                pca = PCA(n_components=n).fit(X_train)
                dt_train_pca = pca.transform(X_train)
                dt_test_pca = pca.transform(X_test)

                modelPCA = model.fit(dt_train_pca, y_train)

                score = modelPCA.score(dt_test_pca, y_test)

                if score > best_score:
                    best_score = score
                    best_numbr = n

            print(f'   SCORE PCA: {best_score} | n = {best_numbr}')

            return modelPCA.predict(dt_test_pca)

        elif solver == 'ipca':
            ipca = IncrementalPCA()
            model_ipca = ipca.fit(X_train)

            print("   Calculating best component and best score for iPCA...")
            best_score = 0
            best_numbr = 0

            for n in np.arange(model_ipca.n_components_) + 1:
                ipca = IncrementalPCA(
                    n_components=n, batch_size=n).fit(X_train)
                dt_train_ipca = ipca.transform(X_train)
                dt_test_ipca = ipca.transform(X_test)

                modeliPCA = model.fit(dt_train_ipca, y_train)

                score = modelPCA.score(dt_test_ipca, y_test)

                if score > best_score:
                    best_score = score
                    best_numbr = n

            print(f'   SCORE iPCA: {best_score} | n = {best_numbr}')

            return modeliPCA.predict(dt_test_ipca)

        elif solver == 'lda':
            print("   Calculating best component and best score for LDA...")
            nComp = min(X_train.shape[1], np.unique(y_train).shape[0] - 1)
            print(f'   The maximum number of components allowed is {nComp}')

            lda = LinearDiscriminantAnalysis(
                n_components=nComp, solver='svd').fit(X_train, y_train)
            dt_train_lda = lda.transform(X_train)
            dt_test_lda = lda.transform(X_test)

            modelf = model.fit(dt_train_lda, y_train)

            print(f'   SCORE LDA: {modelf.score(dt_test_lda, y_test)}')

            return modelf.predict(dt_test_lda)

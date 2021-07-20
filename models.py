from AdaBoost import ABC
from GradientBoosting import GBC
from XGBoost import XGB


class Models:
    def __init__(self, df, data=1, prePros=1, boostTec=1, bestModel=None):
        '''
        Returns the results of using the AdaBoost, Gradient Boosting, and
        XGBoost techniques with PCA, LDA, and no-preprocessing techniques.

        ## Parameters
        df: array-like of shape (n_samples, n_features)
            Dataset to work with. Can be for example a list, or an array.
        data: int, it must match with dataset used in the `df` variable. Options:
            1: innerArm (default)
            2: vein
            3: thumbNail
            4: earLobe
        prePros: int, sets the type of preprocessing to perform. Before using the
            PCA or LDA techniques it is strictly necessary to have obtained the
            same value corresponding to the data 'without preprocessing'. Options:
            1: without preprocessing (default)
            2: PCA preprocessing
            3: LDA preprocessing
        boostTec: int, sets the boosting technique to use. Options:
            1: AdaBoost (default)
            2: Gradient Boosting
            3: XGBoost
        '''
        self.df = df

        if prePros == 1:           # Sin preprocesamiento
            if data == 1:          # innerArm
                if boostTec == 1:   # AdaBoost
                    self.abc_iA = ABC(self.df, 'innerArm', rs=1)
                elif boostTec == 2:  # Gradient Boosting
                    self.gbc_iA = GBC(self.df, 'innerArm', rs=1)
                elif boostTec == 3:  # XGBoost
                    self.xgbc_iA = XGB(self.df, 'innerArm', rs=1)
            elif data == 2:        # vein
                if boostTec == 1:   # AdaBoost
                    self.abc_v = ABC(self.df, 'vein', rs=1)
                elif boostTec == 2:  # Gradient Boosting
                    self.gbc_v = GBC(self.df, 'vein', rs=1)
                elif boostTec == 3:  # XGBoost
                    self.xgbc_v = XGB(self.df, 'vein', rs=1)
            elif data == 3:        # thumbNail
                if boostTec == 1:   # AdaBoost
                    self.abc_tN = ABC(self.df, 'thumbNail', rs=1)
                elif boostTec == 2:  # Gradient Boosting
                    self.gbc_tN = GBC(self.df, 'thumbNail', rs=1)
                elif boostTec == 3:  # XGBoost
                    self.xgbc_tN = XGB(self.df, 'thumbNail', rs=1)
            elif data == 4:        # earLobe
                if boostTec == 1:   # AdaBoost
                    self.abc_eL = ABC(self.df, 'earLobe', rs=1)
                elif boostTec == 2:  # Gradient Boosting
                    self.gbc_eL = GBC(self.df, 'earLobe', rs=1)
                elif boostTec == 3:  # XGBoost
                    self.xgbc_eL = XGB(self.df, 'earLobe', rs=1)
        elif prePros == 2:         # Preprocesamiento PCA
            if data == 1:          # innerArm
                if boostTec == 1:   # AdaBoost
                    abc_iA_pca = ABC(
                        self.df, 'innerArm', rs=1, prePros=True, model=bestModel, solver='pca')
                elif boostTec == 2:  # Gradient Boosting
                    gbc_iA_pca = GBC(
                        self.df, 'innerArm', rs=1, prePros=True, model=bestModel, solver='pca')
                elif boostTec == 3:  # XGBoost
                    xgbc_iA_pca = XGB(
                        self.df, 'innerArm', rs=1, prePros=True, model=bestModel, solver='pca')
            elif data == 2:        # vein
                if boostTec == 1:   # AdaBoost
                    abc_v_pca = ABC(self.df, 'vein', rs=1, prePros=True,
                                    model=bestModel, solver='pca')
                elif boostTec == 2:  # Gradient Boosting
                    gbc_v_pca = GBC(self.df, 'vein', rs=1, prePros=True,
                                    model=bestModel, solver='pca')
                elif boostTec == 3:  # XGBoost
                    xgbc_v_pca = XGB(
                        self.df, 'vein', rs=1, prePros=True, model=bestModel, solver='pca')
            elif data == 3:        # thumbNail
                if boostTec == 1:   # AdaBoost
                    abc_tN_pca = ABC(
                        self.df, 'thumbNail', rs=1, prePros=True, model=bestModel, solver='pca')
                elif boostTec == 2:  # Gradient Boosting
                    gbc_tN_pca = GBC(
                        self.df, 'thumbNail', rs=1, prePros=True, model=bestModel, solver='pca')
                elif boostTec == 3:  # XGBoost
                    xgbc_tN_pca = XGB(
                        self.df, 'thumbNail', rs=1, prePros=True, model=bestModel, solver='pca')
            elif data == 4:        # earLobe
                if boostTec == 1:   # AdaBoost
                    abc_eL_pca = ABC(
                        self.df, 'earLobe', rs=1, prePros=True, model=bestModel, solver='pca')
                elif boostTec == 2:  # Gradient Boosting
                    gbc_eL_pca = GBC(
                        self.df, 'earLobe', rs=1, prePros=True, model=bestModel, solver='pca')
                elif boostTec == 3:  # XGBoost
                    xgbc_eL_pca = XGB(
                        self.df, 'earLobe', rs=1, prePros=True, model=bestModel, solver='pca')
        elif prePros == 3:         # Preprocesamiento LDA
            if data == 1:          # innerArm
                if boostTec == 1:   # AdaBoost
                    abc_iA_lda = ABC(
                        self.df, 'innerArm', rs=1, prePros=True, model=bestModel, solver='lda')
                elif boostTec == 2:  # Gradient Boosting
                    gbc_iA_lda = GBC(
                        self.df, 'innerArm', rs=1, prePros=True, model=bestModel, solver='lda')
                elif boostTec == 3:  # XGBoost
                    xgbc_iA_lda = XGB(
                        self.df, 'innerArm', rs=1, prePros=True, model=bestModel, solver='lda')
            elif data == 2:        # vein
                if boostTec == 1:   # AdaBoost
                    abc_v_lda = ABC(self.df, 'vein', rs=1, prePros=True,
                                    model=bestModel, solver='lda')
                elif boostTec == 2:  # Gradient Boosting
                    gbc_v_lda = GBC(self.df, 'vein', rs=1, prePros=True,
                                    model=bestModel, solver='lda')
                elif boostTec == 3:  # XGBoost
                    xgbc_v_lda = XGB(
                        self.df, 'vein', rs=1, prePros=True, model=bestModel, solver='lda')
            elif data == 3:        # thumbNail
                if boostTec == 1:   # AdaBoost
                    abc_tN_lda = ABC(
                        self.df, 'thumbNail', rs=1, prePros=True, model=bestModel, solver='lda')
                elif boostTec == 2:  # Gradient Boosting
                    gbc_tN_lda = GBC(
                        self.df, 'thumbNail', rs=1, prePros=True, model=bestModel, solver='lda')
                elif boostTec == 3:  # XGBoost
                    xgbc_tN_lda = XGB(
                        self.df, 'thumbNail', rs=1, prePros=True, model=bestModel, solver='lda')
            elif data == 4:        # earLobe
                if boostTec == 1:   # AdaBoost
                    abc_el_lda = ABC(
                        self.df, 'earLobe', rs=1, prePros=True, model=bestModel, solver='lda')
                elif boostTec == 2:  # Gradient Boosting
                    gbc_eL_lda = GBC(
                        self.df, 'earLobe', rs=1, prePros=True, model=bestModel, solver='lda')
                elif boostTec == 3:  # XGBoost
                    xgbc_eL_lda = XGB(
                        self.df, 'earLobe', rs=1, prePros=True, model=bestModel, solver='lda')

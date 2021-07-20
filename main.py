"""
Created by kardantel at 7/20/2021
__author__ = 'Carlos Pimentel'
__email__ = 'carlosdpimenteld@gmail.com'
"""

from models import Models
from utils import Utils

utils = Utils()

if __name__ == "__main__":

    ramanList = utils.load_from_csv('./in', clear=True)

    innerArm = ramanList[3]     # innerArm
    vein = ramanList[1]         # vein
    thumbNail = ramanList[0]    # thumbNail
    earLobe = ramanList[2]      # earLobe

    # Next, you must choose the data to use (data) that must
    # match the variable 'dataset', the type of preprocessing
    # (prePros) and the boosting technique to use (boostTec).
    # The options are as follows:

    # For 'data' you have the options:
    # 1: innerArm
    # 2: vein
    # 3: thumbNail
    # 4: earLobe

    # For 'prePros' you have the options:
    # 1: sin preprocesamiento
    # 2: preprocesamiento PCA
    # 3: preprocesamiento LDA

    # For 'boostTec' you have the options:
    # 1: AdaBoost
    # 2: Gradient Boosting
    # 3: XGBoost

    # For example, if I want to combine the thumbNail dataset
    # with PCA and the AdaBoost technique I would:
    # Models(thumbNail, data=3, prePros=2, boostTec=1)

    # AdaBoost without preprocessing in thumbNail
    result = Models(thumbNail, data=3, prePros=1, boostTec=1)

    # AdaBoost with PCA preprocessing in thumbNail
    Models(thumbNail, data=3, prePros=2, boostTec=1,
           bestModel=result.abc_tN.best_model)

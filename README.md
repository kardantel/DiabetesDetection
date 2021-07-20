# Boosting classifiers for detection of Diabetes Mellitus with Raman spectroscopy

This repository presents the code created as a final Master's project with which a comparative analysis of the results is carried out by using the [AdaBoost](https://www.sciencedirect.com/science/article/pii/S002200009791504X "AdaBoost"), [Gradient Boosting](https://www.sciencedirect.com/science/article/abs/pii/S0167947301000652 "Gradient Boosting") and [XGBoost](https://dl.acm.org/doi/abs/10.1145/2939672.2939785 "XGBoost") boosting techniques, along with PCA and LDA dimensionality reduction techniques, in the dataset [Raman spectroscopy of Diabetes](https://www.kaggle.com/codina/raman-spectroscopy-of-diabetes "Raman spectroscopy of Diabetes"), in order to find the best performing model capable of classifying a patient as diabetic or non-diabetic.

### Features

- Uses the *innerArm*, *vein*, *thumbNail*, and *earLobe* datasets to find the best performing model.
- Use *GridSearchCV* for AdaBoost and *RandomizedSearchCV* for Gradient Boosting and XGBoost;
- For each boosting technique, the best model obtained and its best accuracy are indicated based on the grid of parameters.;
- A metric report is printed for each model and each dataset including, *confusion matrix*, *ROC curves* and *accuracy*;
- `utils.py` contains methods that allow to perform different calculations that help classifiers.

## How to use

In `main.py` you can find the 4 data sets, named above, with which you can perform different combinations for the three boosting techniques and dimensionality reduction techniques.

You must choose the data to use (data) that must match the variable 'dataset', the type of preprocessing (prePros) and the boosting technique to use (boostTec). The options are as follows:

`data`: int, it must match with dataset used in the `df` variable. Options:
1. innerArm (default)
2. vein
3. thumbNail
4. earLobe

`prePros`: int, sets the type of preprocessing to perform. Before using the PCA or LDA techniques it is strictly necessary to have obtained the same value corresponding to the data 'without preprocessing'. Options:
1. without preprocessing (default)
2. PCA preprocessing
3. LDA preprocessing

`boostTec`: int, sets the boosting technique to use. Options:
1. AdaBoost (default)
2. Gradient Boosting
3. XGBoost

For example, if you want to combine the thumbNail dataset with PCA and the AdaBoost technique you would do:

`Models(thumbNail, data=3, prePros=2, boostTec=1)`

### Example

#### AdaBoost without preprocessing in earLobe:

`result = Models(thumbNail, data=4, prePros=1, boostTec=1)`

![](https://i.imgur.com/Vz7hFPt.png)

> AdaBoost - No preprocessing - earLobe

#### AdaBoost with PCA preprocessing in earLobe

`Models(thumbNail, data=4, prePros=2, boostTec=1, bestModel=result.abc_eL.best_model)`

![](https://i.imgur.com/VFdrJ49.png)

> AdaBoost - PCA preprocessing - earLobe

# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
- The Data set used here is from a Banking sector. 

- The main goal of the project is to classify whether a customer will subscribe to a term deposit or not, here 'y' being the dependant variable.

- Please click here to get the Data Set: "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

- The Hyperdrive run model gave the Accuracy of 90.73% for the configuration set & the best model was from AutoML with an accuracy of 91.63%.


## Scikit-learn Pipeline
### The following steps carried out for the Scikit-Learn Pipeline:

 - > Loading of the dataset using the TabularDatasetFactory.
 
 - > This loaded data is preprocessed/cleaned in the training script (train.py), where:
    - Null values are dropped
    - One-hot encoding performed.
    - "y" dependent variable is separated

- > This preprocessed data is split into 80/20 ratio using the SkLearn train_test_split() function.

- > This training data is fit into a LogisticRegression model.

- > HyperdriveCnfiguration is used to tune paramenters: __"C"__ & __"number_of_iterations"__.

- __Hyperparameter Tuning__: Hyperparameter values are randomly selected from the defined search space & also supported by early termination of low performance runs.

- For the project the discrete values for inverse regularization chosen were (0.1, 0.3, 0.5, 1, 10, 50, 100), here the lower value means stronger regularization & the best model had inverse of regularization strength of 0.1

- The discrete values chosen for Max iteration were 50,100,150,200,250. The best model had Max iteration of 150.

**What are the benefits of the parameter sampler you chose?**
- 

**What are the benefits of early stopping policy chosen?**
- The benefit of having "enable_early_stopping=True" is that, the training terminates early if there is no further improvements in the score. 
- The chosen Bandit policy is based on slack factor and evaluation interval. I have defined slackfactor = 0.1, The policy terminates runs where the primary metric is not within the specified slack factor compared to the best performing run.

## AutoML

- The best performing model gave Accuracy of 91.63% with the VotingEnsemble Algorithm, via the AutoML Configurations shown below.

![screenshot](https://github.com/mohammedaz33m/Udacity_Microsoft-Azure_NanoDegree_project/blob/main/ptoject%201/Project%201%20Images/AutoML%20Config.png)

Both AutoML and HyperDrive was configured with same dataset and same primary metric. AutoML was able to train model having the best accuracy. the screenshot below shows models with best accuracy in descending order. VotingEnsemble model outperformed all other models, scoring accuracy of 91.63 %.

- The below image shows the time taken for AutoML & other details

![screenshot](https://github.com/mohammedaz33m/Udacity_Microsoft-Azure_NanoDegree_project/blob/main/ptoject%201/Project%201%20Images/AutoML%20Details.png)

- The below image shows all the models with their accuracies
![screenshot](https://github.com/mohammedaz33m/Udacity_Microsoft-Azure_NanoDegree_project/blob/main/ptoject%201/Project%201%20Images/AutoML%20Models%20run.png)

- VotingEnsemble is an ensemble machine learning model that combines the predictions from multiple models resulting in performance improvement than any single ML models, The VotingEnsamble model trained by Azure AutoML combines models like LigthGBM, XGBoostClassifier, SGD, RandomForest. The VotingEnsemble uses ensemble weight to ensure that better algorithms contribute more to the overall result.


## Pipeline comparison
- The model trained through Azure AutoML outperformed the Logistic regression model that was tuned using Azure HyperDrive. In AutoML some of the best models in descending order by accuracy were VotingEnsemble, MaxAbsScaler,LightGBM, StandardScalerWrapper,XGBoostClassifier. VotingEnsemble had the best accuracy score of 91.63 %. The Logistic regression tuned with HyperDrive had the accuracy of 90.73%.

- Though the difference of the accuracy for model trained with AutoML and HyperDrive was not significant, Azure AutoML made it easy to train multiple models in a short time, with hyperdrive it would have required to create different pipeline for different models.

## Future work
- The data set is quite imbalanced so, it would be quite better to train AutoML & HyperDrive using the F1-Score/AUC, which might improve the results.

- Since the data was preprocessed the AutoML had good score,I can check if un-processed data is fed to AutoML & what would be the output.

- We can also study the impact of inncreasing number of clusters used to study to get faster reults. All these could help us in reducing error in our model and also help us to study the model much faster.


## Proof of cluster clean up
The cluster is deleted via code
![screenshot](https://github.com/mohammedaz33m/Udacity_Microsoft-Azure_NanoDegree_project/blob/main/ptoject%201/Project%201%20Images/Delete%20cluster.png)

Proof that the cluster deletion was initiated & deleted
![screenshot](https://github.com/mohammedaz33m/Udacity_Microsoft-Azure_NanoDegree_project/blob/main/ptoject%201/Project%201%20Images/Deleting%20Cluster.png)

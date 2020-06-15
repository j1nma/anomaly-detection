## Anomaly Detection
<img src="https://github.com/j1nma/anomaly-detection/blob/master/docs/images/XGB_confusion_matrix.png?raw=true" width="350"/>  <img src="https://github.com/j1nma/anomaly-detection/blob/master/docs/images/roc-auc-curve.png?raw=true" width="400"/>

This work does experiments with anomaly / outlier detection methods to find the fraudulent cases. 

Various classifier is utilized, with sampling methods to help the classifier learn the minority class better, and also using a higher cost associated with misclassification of the fraudalent cases. Fitting criteria for evaluation was chosen.

Finally, the effectiveness of all the methods tried is compared, and an analyse on which one might be used in a real setting is done.

### Installation

### Running

Custom hyperparameters in a textfile i.e. _"./configs/config.txt"_.


A _results_ folder will contain a timestamp directory with the latest results.

### Datasets
* The credit card fraud dataset, available from e.g. Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud ;
* The IEEE Fraud detection dataset at https://www.kaggle.com/c/ieee-fraud-detection

A ```data``` folder must be provided by the user with ```/creditcard``` and ```/ieee``` subfolders with their
corresponding files, i.e. ```/data/ieee/train_transaction.csv```.

### Techniques
* Unsupervised: Isolation Forest, Local Outlier Factor, an AutoEncoder and One-Class Support Vector Machine.
* Supervised: XGBClassifier.
* Semi-supervised: Gaussian Mixture.

### Report
Anomaly Detection - Cavallin, Alonso.pdf
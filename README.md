## Anomaly Reduction
This work does experiments with anomaly / outlier detection methods to find the fraudulent cases. 

A method based on a two-class classifier is included, with sampling methods to help the classifier learn the minority class better, and also using a higher cost associated with misclassification of the fraudalent cases. Fitting criteria for evaluation was chosen.

3 methods specifically tailored to anomaly detection are used.

Finally, the effectiveness of all the methods tried is compared, and an analyse on which one might be used in a real setting is done.

### Installation

### Running

Custom hyperparameters in a textfile i.e. _"./configs/config.txt"_.


A _results_ folder will contain a timestamp directory with the latest results.

### Datasets
* The credit card fraud dataset, available from e.g. Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud ;
* The IEEE Fraud detection dataset at https://www.kaggle.com/c/ieee-fraud-detection

### Techniques
3 methods specifically tailored to anomaly detection, e.g. one-class classifiers, statistical/density based methods, ... You can e.g. use the methods provided in sk-learn as an inspiration, but there are also other approaches published that can be re-used.
* TODO
* TODO
* TODO

### Report
Anomaly-Detection-Alonso.pdf
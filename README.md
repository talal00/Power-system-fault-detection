# Power-system-fault-detection

In this code I have implemented an Artificial Neural Network based power system line faults detection. In this notebook, a Multi-layer perceptron (MLP) Classifier is implemented which uses Backpropagation. The data used is a time series data which was generated in MATLAb. Two target values 0 and 1 are used to create binary classes. When 0 is identified then there is no fault and 1 shows fault is detected. The data consist of different features and only six features (3-phase current and voltage measurements are used to train the MLP model). The code can be divided into following steps

1. Data is read, preprocessed, and statistically analyzed.

2. After pre-processing the data is splitted into test, train, and predict datasets

3. MLP classifier model is trained with trainning dataset, tested, and predicted with seperated data which was not used in training process.

4. Results of MLP classifier

5. Data Visualization


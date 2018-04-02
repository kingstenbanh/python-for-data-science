from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the data
telescope = pd.read_csv('MAGIC Gamma Telescope Data.csv')

# Clean the data
telescope_shuffle = telescope.iloc[np.random.permutation(len(telescope))]
tele = telescope_shuffle.reset_index(drop=True)

# Store 2 classes
tele['Class'] = tele['Class'].map({ 'g': 0, 'h': 1 })
tele_class = tele['Class'].values

# Split training, testing, and validation data
training_indices, validation_indices = training_indices, testing_indices = train_test_split(tele.index, 
    stratify = tele_class, train_size=0.75, test_size=0.25)

# Let Genetic Programming find the best ML model and hyperparameters
tpot = TPOTClassifier(generations=5, verbosity=2)
tpot.fit(tele.drop('Class', axis=1).loc[training_indices].values,
    tele.loc[training_indices, 'Class'].values)

# Score the accuracy
tpot.score(tele.drop('Class', axis=1).loc[validation_indices].values,
    tele.loc[validation_indices, 'Class'].values)

# Export the generated code
tpot.export('pipeline.py')

GradientBoostingClassifier(
    BernoulliNB(MaxAbsScaler(input_matrix), alpha=0.001, fit_prior=True), 
    learning_rate=0.1, max_depth=9, max_features=0.55, min_samples_leaf=20, 
    min_samples_split=14, n_estimators=100, subsample=0.7
)

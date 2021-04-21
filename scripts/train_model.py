
### SETUP ###

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

plt.interactive(False)

PATH_DATA_PROCESSED = Path('data/processed')
PATH_MODELS = Path('models')
DATASET_NAME = 'dataset_2021_03_01_wins_only.csv'

### LOAD AND SPLIT DATA ###

df = pd.read_csv(PATH_DATA_PROCESSED/DATASET_NAME)
seasons = df['season'].unique()
validation_seasons = np.sort(seasons)[-1:] ## use the last season as a validation set
train_seasons = [season for season in seasons if season not in validation_seasons]

df_train = df[df['season'].isin(train_seasons)]
df_valid = df[df['season'].isin(validation_seasons)]

X_vars = ['wins_per_game_home', 'wins_per_game_away']
y_var = 'home_win'

X_train = df_train[X_vars]; X_valid = df_valid[X_vars]
y_train = df_train[y_var]; y_valid = df_valid[y_var]


### TRAIN MODEL ###

lr = LogisticRegression()
lr.fit(X_train, y_train)

### PERFORMANCE ###
coefficients = pd.DataFrame({'variable':X_train.columns, 'coefficient':lr.coef_[0]})

probs_train = lr.predict_proba(X_train)
probs_val = lr.predict_proba(X_valid)

plt.hist(probs_val[:,1])
plt.show()

## set cut-off so that prob home win roughly correct in train set
np.mean(y_train)
np.mean(probs_train[:,1] > 0.539)
cutoff = 0.539

print(metrics.classification_report(y_train, probs_train[:,1] > cutoff))
print(metrics.classification_report(y_valid, probs_val[:,1] > cutoff))

metrics.confusion_matrix(y_valid, probs_val[:,1] > cutoff)

pickle.dump(lr, open('models/2021_04_20_logreg_win_percentage_only.pickle', 'wb'))


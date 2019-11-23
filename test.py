"""
This script is for debugging the library in an IDE
"""

import numpy as np
import sklearn.datasets
#import matplotlib.pyplot as plt
import itertools

import linear_classifier  # librairie du devoir ift603-712
import utils   # librairie du devoir ift603-712

np.random.seed(0)

# DATA ====================================================
iris_dataset = sklearn.datasets.load_iris()
X_complete = iris_dataset.data
y_ = iris_dataset.target
feature_names = iris_dataset.feature_names
num_classes = 3

print('Données    X  ' + str(X_complete.shape))
print('Étiquettes y  ' + str(y_.shape))
print('Nombre de classes: ' + str(num_classes))
print('Nom des features: ' + str(feature_names))

# Ajouter du bruit pour rendre la tâche plus difficile
coef_bruit = 0.0
mean = np.mean(X_complete, axis=0)
std = np.std(X_complete, axis=0)
X_complete += coef_bruit * std * np.random.randn(*X_complete.shape)

# Indiquez vos deux choix ici, parmi (0, 1, 2, 3):
d1 = 0  # Dimension 1
d2 = 3  # Dimension 2
X_ = X_complete[:, (d1, d2)]  # On ne garde que d1 et d2
print('Caractéristiques choisies: "{}" et "{}"'.format(feature_names[d1], feature_names[d2]))

# Centrer et réduire les données (moyenne = 0, écart-type = 1)
mean = np.mean(X_, axis=0)
std = np.std(X_, axis=0)
X_ = (X_ - mean) / std

num_train = 75
num_val = 30
num_test = 45

idx = np.random.permutation(len(X_))

train_idx = idx[:num_train]
val_idx = idx[num_train:num_train + num_val]
test_idx = idx[-num_test:]

X_train = X_[train_idx]
y_train = y_[train_idx]
X_val = X_[val_idx]
y_val = y_[val_idx]
X_test = X_[test_idx]
y_test = y_[test_idx]


accu = utils.test_sklearn_svm(X_train, y_train, X_test, y_test)
print('Test accuracy: {:.3f}'.format(accu))
if accu < 0.8:
    print('ERREUR: L\'accuracy est trop faible. Il y a un problème avec les données.')
    
model = linear_classifier.LinearClassifier(X_train, y_train, X_val, y_val, num_classes=3, bias=True)

loss_train_curve, loss_val_curve, accu_train_curve, accu_val_curve = model.train(num_epochs=15, l2_reg=0.0)

print('[Training]   Loss: {:.3f}   Accuracy: {:.3f}'.format(loss_train_curve[-1], accu_train_curve[-1]))
print('[Validation] Loss: {:.3f}   Accuracy: {:.3f}'.format(loss_val_curve[-1], accu_val_curve[-1]))
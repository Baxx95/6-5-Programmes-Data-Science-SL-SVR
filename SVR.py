# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv('age.csv')

data.columns

data = data.drop(['Sexe'], axis=1)

#on recupere tout sauf la colonne Age
X = data.iloc[:,:-1].values

y = data['Age'].values

# On a le Kernel (noyau) lineaire, polynomial, gaussien ...
regressor = SVR(kernel='linear', degree=1) # la param degree est optionnel

# on visualise la distribution "du poids ecaillé" = f(age)
plt.scatter(data['Poids écaillé'], data['Age'])

# On partitionne nos données
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .2)

# On entraine notre modèle
regressor.fit(x_train, y_train)

# On fait une prédiction avec le x_test
pred = regressor.predict(x_test)

#================ VERIFIONS L'EXACTITUDE =====================

# Note : score() nous donne des précision de la prédiction
print('Score de prediction : ',  str(regressor.score(x_test, y_test)*100)[0:5],'%')

# Note : r2_score() nous donnes aussi une precision sur la prediction
print(str(r2_score(y_test, pred)*100)[0:5], '%')



#============= Nous allons créer un deuxieme modèle pour pouvoir les croiser ============
regressor = SVR(kernel='rbf', epsilon=1.0) # la valeur designe le noyau gaussien
regressor.fit(x_train, y_train)
pred_gauss = regressor.predict(x_test)


print('Score() de prediction : ',  str(regressor.score(x_test, y_test)*100)[0:5],'%')
print('r2_score() de prediction : ', str(r2_score(y_test, pred_gauss)*100)[0:5], '%')


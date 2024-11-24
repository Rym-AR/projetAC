import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


chemin_fichier = r'C:\\Users\\Cli\\Downloads\\cars.csv'  # Remplacer par le chemin du fichier Salary_Data.csv
cars = pd.read_csv(chemin_fichier)
copy= cars
'''----------------les valeurs uniques de chaque feature---------------------'''
Brand=cars['Brand'].unique()
Price=cars['Price'].unique()
Body= cars['Body'].unique()
Mileage= cars['Mileage'].unique()
EngineV=cars['EngineV'].unique()
EngineType= cars['Engine Type'].unique()
Registration= cars['Registration'].unique()
Year= cars['Year'].unique()
Model=cars['Model'].unique()
'''-------------- Visualiser les features ---------------------'''
value_counts = copy['Brand'].value_counts()
# Créer le diagramme en barres
plt.bar(value_counts.index, value_counts)
plt.xlabel('Valeurs de Brand')
plt.ylabel('Fréquence')
plt.title('Fréquence des valeurs de Brand')
plt.show()

value_counts = copy['Body'].value_counts()
# Créer le diagramme en barres
plt.bar(value_counts.index, value_counts)
plt.xlabel('Valeurs de Body')
plt.ylabel('Fréquence')
plt.title('Fréquence des valeurs de Body')
plt.show()
# Compter les occurrences de chaque valeur unique dans la colonne engineV
value_counts = copy['EngineV'].value_counts()

# Créer le diagramme en barres
plt.bar(value_counts.index, value_counts)
plt.xlabel('Valeurs de engineV')
plt.ylabel('Fréquence')
plt.title('Fréquence des valeurs de engineV')
plt.show()

value_counts = copy['Mileage'].value_counts()
# Créer le diagramme en barres
plt.bar(value_counts.index, value_counts)
plt.xlabel('Valeurs de Mileage')
plt.ylabel('Fréquence')
plt.title('Fréquence des valeurs de Mileage')
plt.show()

value_counts = copy['Engine Type'].value_counts()
# Créer le diagramme en barres
plt.bar(value_counts.index, value_counts)
plt.xlabel('Valeurs de Engine Type')
plt.ylabel('Fréquence')
plt.title('Fréquence des valeurs de Engine Type')
plt.show()

value_counts = copy['Registration'].value_counts()
# Créer le diagramme en barres
plt.bar(value_counts.index, value_counts)
plt.xlabel('Valeurs de Registration')
plt.ylabel('Fréquence')
plt.title('Fréquence des valeurs de Registration')
plt.show()

value_counts = copy['Year'].value_counts()
# Créer le diagramme en barres
plt.bar(value_counts.index, value_counts)
plt.xlabel('Valeurs de Year')
plt.ylabel('Fréquence')
plt.title('Fréquence des valeurs de Year')
plt.show()

value_counts = copy['Model'].value_counts()
# Créer le diagramme en barres
plt.bar(value_counts.index, value_counts)
plt.xlabel('Valeurs de Model')
plt.ylabel('Fréquence')
plt.title('Fréquence des valeurs de Model')
plt.show()
'''-----------graphe boite a moustache (outliers) de tous le dataset-------------'''
fig = plt.figure(figsize=(100,150))
copy.plot.box()

'''--------------------------diviser X et y-------------------------------'''
features= ['Body', 'Brand', 'Mileage', 'EngineV', 'Engine Type', 'Registration', 'Year', 'Model']
X = copy[features]
y=copy['Price']
'''----------------------------------Encodage----------------------------'''
from sklearn.preprocessing import LabelEncoder
# Création d'un objet LabelEncoder
le = LabelEncoder()
# Encodage de la colonne 'couleur'
X['Brand'] = le.fit_transform(X['Brand'])
X['Body'] = le.fit_transform(X['Body'])
X['Engine Type'] = le.fit_transform(X['Engine Type'])
X['Registration'] = le.fit_transform(X['Registration'])
X['Model'] = le.fit_transform(X['Model'])
'''------------------------------echantillonage---------------------------'''
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
'''--------------------gestion des valeurs manquantes--------------------'''    
#les noms des colonnes qui contiennent des valeurs manquantes
print(cars.columns[cars.isnull().any()])
#nombre de valeurs manquantes dans la colonne EngineV
nan_engineV_count = cars['EngineV'].isnull().sum()
print("Nombre de valeurs manquantes dans la colonne 'EngineV':",nan_engineV_count)
#nombre de valeurs manquantes dans la colonne Price
nan_Price_count = cars['Price'].isnull().sum()
print("Nombre de valeurs manquantes dans la colonne 'Price':",nan_Price_count)
'''-------------calculer moyenne avec les boucles(sans fonction predefinis)-------'''
# Calcul de la moyenne (sans utiliser df.mean())
total = 0
count = 0
for value in X['EngineV']:
   if not np.isnan(value):
        total += value
        count += 1
mean = total / count
moyenne_EngineV = np.round(mean, 1)
print('la moyenne est ',moyenne_EngineV)

# Remplacement des NaN par la moyenne
X_train.loc[X_train['EngineV'].isnull(), 'EngineV'] = moyenne_EngineV
X_test.loc[X_test['EngineV'].isnull(), 'EngineV'] = moyenne_EngineV

'''-----------------supprimer les valeurs manquantes de la colonne Price-----------'''

y = pd.DataFrame(y)
y=y.dropna()
#les noms des colonnes qui contiennent des valeurs manquantes
print('les colonnes des valeurs manquantes dans copy:',copy.columns[copy.isnull().any()])
print('il n ya aucune valeur manquante dans notre dataframe')


'''----------------------------matrice de correaltion---------------------'''
corr_matrix = X.corr()
# Affichage de la matrice
print(corr_matrix)

'''-----------normalisation (sans fonction predefinis)------------'''
# Min-Max scaling
def min_max_scaling(X_train):
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    return (X_train - min_val) / (max_val - min_val)
# Application des fonctions
X_train = min_max_scaling(X_train)


def min_max_scaling(X_test):
    min_val = np.min(X_test, axis=0)
    max_val = np.max(X_test, axis=0)
    return (X_test - min_val) / (max_val - min_val)
# Application des fonctions
X_test = min_max_scaling(X_test)

fig = plt.figure(figsize=(100,150))
X.plot.box()


''''-----------------Eliminer les outliers de EngineV------------------'''
#remplacer les outliers de EngineV par Q1  et Q3
#visualisation de tous X_Train avant
sns.boxplot(data=X_train)
plt.title('Boîte à moustaches de X_train avaaannnt')
plt.show()
#visualisation de tous X_Test avant
sns.boxplot(data=X_test)
plt.xlabel('Valeurs ')
plt.ylabel('Fréquence')
plt.title('Boîte à moustaches de X_test avaaannnt')
plt.show()
# Détection des outliers avec l'IQR
Q1 = X_train['EngineV'].quantile(0.25)
Q3 = X_train['EngineV'].quantile(0.75)
IQR = Q3 - Q1
# Définir un seuil pour les outliers
threshold = 1.5

sns.boxplot( data=X_train['EngineV'])
plt.xlabel('Valeurs de engineV')
plt.ylabel('Fréquence')
plt.title('avaaanttt Boîte à moustaches de engineV')
plt.show()

# Identifier les outliers de Engine V de X_train et les remplacer par Q1 etQ3
X_train['EngineV'] = X_train['EngineV'].apply(lambda x: Q1 if x < Q1 - threshold*IQR else (Q3 if x > Q3 + threshold*IQR else x))

#boite a moustache de EngineV de X_train apres l'elimination des outliers
sns.boxplot(x=X_train['EngineV'])
plt.xlabel('Valeurs de engineV')
plt.ylabel('Fréquence')
plt.title('nvl Boîte à moustaches de engineV')
plt.show()

# Identifier les outliers de Engine V de X_test et les remplacer par Q1 etQ3
X_test['EngineV'] = X_test['EngineV'].apply(lambda x: Q1 if x < Q1 - threshold*IQR else (Q3 if x > Q3 + threshold*IQR else x))

#boite a moustache de EngineV de X_train apres l'elimination des outliers
sns.boxplot(x=X_test['EngineV'])
plt.xlabel('Valeurs de engineV de X_test apres')
plt.ylabel('Fréquence')
plt.title('nvl Boîte à moustaches de engineV')
plt.show()

''''-----------------Eliminer les outliers de Mileage------------------'''

''''-----------------Eliminer les outliers de Year------------------'''

''''----------------Visualisation apes l'elimination des outliers--------------'''
#visualisation de tous X_Train apres l'elimination des outliers
sns.boxplot(data=X_train)
plt.xlabel('Valeurs ')
plt.ylabel('Fréquence')
plt.title('Boîte à moustaches de X_train apres')
plt.show()

#visualisation de tous X_Test apres l'elimination des outliers
sns.boxplot(data=X_test)
plt.xlabel('Valeurs ')
plt.ylabel('Fréquence')
plt.title('Boîte à moustaches de X_test apreeees')
plt.show()












   

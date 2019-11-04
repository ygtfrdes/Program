# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:13:33 2019

@author: carlyna.bondiombouy
"""
#Change environment 
   #https://stackoverflow.com/questions/32565302/python-after-installing-anaconda-how-to-import-pandas
   #http://docs.continuum.io/anaconda/user-guide/tasks/integration/python-path/
   #https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

#####################################################
#####################################################

#*************************************
########data preprocessing############
#*************************************

######################################
#import libraries :> cmd + enter
######################################

#maths libraries 
import numpy as np 

#visualization libraries
#pyplot modules for 2D graph
import matplotlib.pyplot as plt 

#import dataset, manage dataset
import pandas  as pd 

######################################
#import dataset, before doing this step it's mandatory to choose the good work directory
######################################

##import dataset 

dataset =pd.read_csv('Data.csv')
dataset

#Country, Age, Salary independant variables: on se sert pour predire (variables predictives)
#Purchased are dependant variables : le resultat (variable qu'on doit prédire)
#predict if yes or not the new product


##creer une matrice des variables indépendantes
#iloc retrieve the useful indices of the dataset
#Column 1 variable independante, column2 variable dépendante

X=dataset.iloc[:, :-1].values #country,Age, Salary 
y=dataset.iloc[:, -1].values  #purchased

######################################
#gerer les données manquantes
######################################

#Dans l'exemple il y'a 2 données manquantes (nan), une dans la colonne age et l'autre dans la colonne salaire
#cela dépend de la distribution des datasets
#bien distribué dans trop d'outlayer il faut la distrivuer par la moyenne de la colonne. Sinon si elle n'est pa normalement distribuée avec les gros outlayers il faut remplacer les données manquantes par la médiane de la colonne. parce que les gros outlayers vont provoquer un biais 

#récuper la classe Imputer du module sklearn de Machine learning. Une classe contient des parametres. l'objet imputer

from sklearn.preprocessing import Imputer #ctrl + i for help
imputer= Imputer(missing_values='NaN', strategy='mean', axis=0) #axis=0 pour la moyenne de la colonne. axis=1 pour la moyenne de la ligne
#Imputation transformer for completing missing values.
imputer.fit(X[:, 1:3]) #utiliser une méthode ou une classe sur un objet: objet.nom méthode
                       #imputer.fit(x[:, [1,2]]) in order to take columns one and two. 
                       #visualize nan : np.set_printoptions(threshold=np.nan)
X[:, 1:3] = imputer.transform(X[:, 1:3])
#type(X)

######################################
#gerer les variables catégoriques
######################################

#purchase et country sont des variables manquantes. Parce que les valeurs de ces variables ne sont pas numériques continues
#Ici Country est à 3 catégories: France, Germany et Spain. Purchase est une variable catégorique avec 2 catégorie oui ou non. Age et Salary sont des variables numériques sui prennent des valeurs continues et numériques.
#Gérer les variables catégoriques permet d'éviter d'avoir les problèmes d'implémentation dans une équation mathématique. Donc il faut encoder ces catégories écritent en forme de texte. Il faut encoder en valeur numérique afin que cela rentre dans l'équation.
#On ne peut pas encoder France en donnant la variable 0, Spain par 1 et Germany par 2. Parce qu'il y'a aucun lien entre les 3 sinon lors des traitements on peut avoir des erreurs.
# cela evite les relations d'ordres, car 0 est inferieur à 1 et 1 à 2. Dans la réalité on a aucune relation entre les différents pays. cela s'appelle "One-Hot encoding ou encodage par démi variable
#https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/........A “place” variable with the values: “first”, “second” and “third“. does have a natural ordering of values. This type of categorical variable is called an ordinal variable.
#Pucrchased sera codé par 0 et 1 parce que c'est une variable dépendante

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() #encode country
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])#indice 0 parce que c'est le premier element du dataset # labelEncoder transforme le texte en numérique 
onehotencoder = OneHotEncoder(categorical_features=[0])
X= onehotencoder.fit_transform(X).toarray()#transforme la colonne de x en france, germany, spain et les fit aussi #Encode sous forme de démi variable

labelencoder_y = LabelEncoder() #encode Purchased
y = labelencoder_y.fit_transform(y) #y n'a qu'une colonne 


######################################
#Diviser le dataset entre le training set et le test set 
######################################
#Cette step va nous enable de construire un modèle de ML, et cette step est compulsory. On doit faire cette step, parce que le modèle prend des corrélations des variables indépendantes et dépendantes, il va le faire dans le sous dataset du dataset originale. ce sous dataset c'est le training set (c'est là ou le modele va faire son training il va apprendre des corrélations). Mais pour vérifier qu'il n'y a pas de suraprentissage c'est à dire qu'il n'est pas appris par coeur les corrélations on doit aussi avoir un test set qui va etre que 20 pourcent du dataset originale alors que le training set va répresenter 80 pourcent. On va tester le modèle sur les observations du test set. Car les observations du test set vont etre des nouvelles observations. Les observations sur lesquelles le modèle n'a pas été construit.   

from sklearn.model_selection import train_test_split   #train_test_split est une fonction du module model_selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)# la fonction reroutrne ses 4 entités

#x_train=matrice des variables indépendantes du training set, x_test= matrice des variables indépendantes du test set 
#y_train= vecteur variable dépendante du training test, y_test= vecteur variable dependante du test set
#test_size correspond à la proportion d'observation du dataset qu'on veut mettre dans le test set. ceux sont des nouvelles observations par rapport aux observations dans le training set) 
#randomset c'est juste pour avoir le meme resultat que le prof. ce n'est pas obligatoire
#On construit le modèle et on lui fait apprendre les corrélations sur le training set. Et on évalue ses performances sur le test set.

######################################
#Appliquer feature scaling 
######################################
#Ce step permet de mettre toutes les variables sur la meme echelle, afin qu'aucune  variable n'écrase l'autre dans les equations de machine learning
#Age(valeurs entre 35 et 50 ans) et salaire (valeurs entre 68000 et 72000) n'ont pas les memes valeurs. Cela presente des risques dans les equations de machine learning, parce que le salaire peut écraser la variable age, de sorte que la variable age ne soit plus prise en compte dans le modèle.Cela peut avoir un impact sur la variable dépendante purchased.
#nous utiliserons la classe: 

from sklearn.preprocessing import StandardScaler #la classe 
sc = StandardScaler() # crée l'objet de la classe
X_train = sc.fit_transform(X_train)# on lie l'objet à la matrice de features de  training set puis la scalée
X_test = sc.transform(X_test)#le training set a une distribution similaire du test set voila  pourquoi nous appliquons le feature scaling sur le test set 


#####################################################
#####################################################

#*************************************
########Régresion############
#*************************************
#Les modèles de Régression sont utilisés pour prédire une valeur réelle continue, comme par exemple le salaire. Si votre variable indépendante est le temps, alors votre modèle prédit des valeurs futures. Sinon, votre modèle prédit des valeurs présentes, mais inconnues.
#Régression linéaire simple: équation d'une droite dans le plan a 2 dimension : Y= b0 + b1 * X1
######################################
#import libraries :> cmd + enter
######################################
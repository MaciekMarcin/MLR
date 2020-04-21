# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 08:48:01 2020

@author: Tomek
"""
X = []
y = []

import csv
import numpy as np
with open('C:/Users/Tomek/Desktop/wazne/wykladIO/cwiczenia/diabetes.csv','r') as newFile:
    plots = csv.reader(newFile, delimiter=',')
    has_header = csv.Sniffer().has_header(newFile.read(1024))
    newFile.seek(0) #na poczatek
    if has_header:
        next(plots) 
    for row in plots:
        #print(row)
        X.append(row[0:8])
        y.append(row[8])
        
print(y)        
        
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1, stratify=y)

print('Liczba etykiet w zbiorze y:', np.bincount(y))
print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))

# Standaryzacja cech:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Klasyfikacja KNN
#from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors=11, 
 #                         p=2, #parametr z metryki
  #                        metric='minkowski')
#knn.fit(X_train_std, y_train)
#Ocena
#print(knn.score(X_test_std,y_test))
#y_pred=knn.predict(X_test_std)
#from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_test, y_pred))

#Klasyfikacja DD
#from sklearn.tree import DecisionTreeClassifier
#treeD = DecisionTreeClassifier(criterion='gini', 
 #                             max_depth=5, 
  #                            random_state=1)
#treeD.fit(X_train, y_train)
#from sklearn import tree
#tree.plot_tree(treeD)
#conda install python-graphviz, by zapisac do pdf drzewo i je ogladac
#import graphviz 
#dot_data = tree.export_graphviz(treeD, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("diabetes") 

#dot_data = tree.export_graphviz(treeD, out_file=None,  
                      #feature_names=plots.feature_names,  
                      #class_names=plots.target_names,            
 #                     filled=True, rounded=True,  
  #                    special_characters=True)  
#graph = graphviz.Source(dot_data)  
#graph 


#Ocena
#print(treeD.score(X_test,y_test))
#y_pred = treeD.predict(X_test)
#from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_test, y_pred))

#Klasyfikacja NB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_std, y_train)
#Ocena
print(gnb.score(X_test_std,y_test))
y_pred = gnb.predict(X_test_std)
print("Liczba zle przyporzadkowanych probek sposrod wszystkich %d probek : %d"       
      % (X_test_std.shape[0], (y_test != y_pred).sum()))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))











# coding: utf-8

# *Python. Uczenie maszynowe. Wydanie drugie*, [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Repozytorium kodu: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Licencja: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python. Uczenie maszynowe - kod źródłowy

# # Rozdział 3. Stosowanie klasyfikatorów uczenia maszynowego za pomocą biblioteki scikit-learn

# # Pierwsze kroki z biblioteką scikit-learn — uczenie perceptronu

# Wczytujemy zestaw danych Iris z biblioteki scikit-learn. Trzecia kolumna reprezentuje tutaj długość płatka, a czwarta - jego szerokość. Klasy są przekształcone do postaci liczb całkowitych, gdzie etykieta 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.


from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data
print(X)
X = iris.data[:, [2, 3]]
print(X)
y = iris.target

print('Etykiety klas:', np.unique(y))


# Dzielimy dane na 70% próbek uczących i 30% próbek testowych:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
# if variable y is a binary categorical variable with values 0 and 1 
# and there are 25% of zeros and 75% of ones, stratify=y will make sure 
# that your random split has 25% of 0's and 75% of 1's.

# If you don't mention the random_state in the code, then whenever 
# you execute your code a new random value is generated and the 
# train and test datasets would have different values each time.

# However, if you use a particular value for random_state 
# (random_state = 1 or any other value) everytime the result will 
# be same,i.e, same values in train and test datasets.
# często spotyka się wartosc 42

print('Liczba etykiet w zbiorze y:', np.bincount(y))
print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))


# Standaryzacja cech:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#układa odpowiednio dane do wykresu (pionowo w rzędzie)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test)) #poziomo pod względem kolumn

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # konfiguruje generator znaczników i mapę kolorów
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # rysuje wykres powierzchni decyzyjnej
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # zaznacza przykłady testowe
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='Zestaw testowy')

#uczenie drzewa decyzyjnego

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel('Długość płatka [cm]')
plt.ylabel('Szerokość płatka [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# # Algorytm k-najbliższych sąsiadów — model leniwego uczenia
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, #parametr z metryki
                           metric='minkowski')
knn.fit(X_train_std, y_train)

#print(knn.predict([[0.1,0.1]]))
from sklearn.metrics import accuracy_score

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))
#zakreslone w kolko testowe dane
plt.xlabel('Długość płatka [standaryzowana]')
plt.ylabel('Szerokość płatka [standaryzowana]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()



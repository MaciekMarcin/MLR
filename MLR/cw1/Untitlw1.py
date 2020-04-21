import numpy as np
from scipy.spatial import distance
from functools import reduce
import operator
import random
import statistics
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt

#1.1#####################
def policz(a,b):
    print(a*b)

a = 123
b = 321

policz(a,b)
#########################

list1 = [3,8,9,10,12]
list2 = [8,7,7,5,6]
list3 = []
list4 = []

for i in range(0, len(list1)):
    list3.append(list1[i] + list2[2])

for i in range(0, len(list1)):
    list4.append(list1[i] * list2[2])

print(list3)
print(list4)

#1.2######################

print(sum(list4))

dst = []
dst = distance.euclidean(list1, list2)
#or i in range(0, len(list1)):
#    dst.np.append(np.linalg.norm(list1[i] - list2[2]))
print(dst)

##########################

#1.3######################

A_M = [[1,2,3],[2,3,1],[3,1,2]]
B_M = [[4,5,6],[5,6,4],[6,4,5]]
#result = [[0,0,0],[0,0,0],[0,0,0]]

#for i in range(len(A_M)):
#   for j in range(len(B_M[0])):
#       for k in range(len(B_M)):
#           result[i][j] = A_M[i][k] * B_M[k][j]
#
#for r in result:
#   print(r)

pr = np.matmul(A_M,B_M)
print(pr)

##########################

#1.4######################

list_random = []

for i in range(0,50):
    list_random.append(random.randint(0,100))

print(len(list_random))
print(list_random)

##########################
#1.5######################

print('srednia ' + str(statistics.mean(list_random)))
print('max: ' + str(max(list_random)))
print('min: ' + str(min(list_random)))
print('odchylenie: ' + str(math.sqrt(statistics.variance(list_random))))

##########################

#1.6######################

#def normalize(x):
#    return((x - min(x))/(max(x) - min(x)))

##########################

#2.1######################

x = []
y = []
z = []
t = []

#with open('miasta.csv', newline='') as File:  
#    reader = csv.reader(File)
#    for row in reader:
#        print(row)

#with open('miasta.csv', 'a', newline='') as newFile:
#    newFileWriter = csv.writer(newFile)
#    newFileWriter.writerow([2010,460,555,405])

with open('miasta.csv', 'r') as File:  
    #reader = csv.reader(File)
    plots = csv.reader(File, delimiter=',')
    has_header = csv.Sniffer().has_header(File.read(1024))
    File.seek(0)
    if has_header:
        next(plots)
    for row in plots:
        print(row)
        x.append(int(row[0]))
        y.append(int(row[1]))
        z.append(int(row[2]))
        t.append(int(row[3]))

#df = pd.read_csv('miasta.csv')
#print(df)

plt.plot(x,y, color='r')
plt.plot(x,z, color='b')
plt.plot(x,t, color='y')
plt.legend(('Gdansk','Poznan','Szczecin'))
plt.xlabel('Lata')
plt.ylabel('Liczba ludnosci')
plt.title('Liczba ludnosci w latach')
plt.show()

##########################
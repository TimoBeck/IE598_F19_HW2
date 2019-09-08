#Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

#Import Treasury Squeeze dataset 
df = pd.read_csv('treasury_squeeze.csv',header=None)

# Extract data to form trainning and test sets 
# Last col of df is the target
X = df.iloc[1:,2:11]
# Extract target
y = df.iloc[1:,11]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 21, stratify=y)

k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))

print("Accuracy using KNeighborsClassifier is at the highest when k = " + str(scores.index(max(scores))+1))
print(max(scores))

plt.plot(k_range,scores,'o')
plt.ylabel('accuracy score')
plt.xlabel('n_neighbors = k')
plt.title('K-neareast neighbors')

# Decision Tree
# Decision Tree
l_range = range(1,26)
scores_2 = []
for l in l_range:
    dt = DecisionTreeClassifier(max_depth = l, random_state = 1)
    dt.fit(X_train,y_train)
    y_pred_tree = dt.predict(X_test)
    scores_2.append(accuracy_score(y_test,y_pred_tree))
    
print("Accuracy using DecisionTreeClassifier is at the highest when l = " + str(scores_2.index(max(scores_2))+1))
print(max(scores_2))

plt.figure()
plt.plot(l_range,scores_2,'o')
plt.ylabel('accuracy score')
plt.xlabel('Max_depth = l')
plt.title('Decision tree')

print("My name is Timothee Becker")
print("My NetID is: tbecker5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


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

# Decision Tree
dt = DecisionTreeClassifier(max_depth = 2, random_state = 1)
dt.fit(X_train,y_train)
y_pred_tree = dt.predict(X_test)
print("Accuracy using DecisionTreeClassifier is: " + str(accuracy_score(y_test, y_pred)))

print("My name is Timothee Becker")
print("My NetID is: tbecker5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


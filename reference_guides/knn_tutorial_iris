
#Import the load_iris function from datsets module
from sklearn.datasets import load_iris
#import the KNeighborsClassifier class from sklearn
from sklearn.neighbors import KNeighborsClassifier
#import metrics model to check the accuracy
from sklearn import metrics


#Create bunch object containing iris dataset and its attributes.
iris = load_iris()


#Print the iris data
iris.data

#Names of 4 features (column names)
print(iris.feature_names)

#Integers representing the species: 0 = setosa, 1=versicolor, 2=virginica
print(iris.target)

# Feature matrix in a object named X
X = iris.data
# response vector in a object named y
y = iris.target


#Try running from k=1 through 25 and record testing accuracy
k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
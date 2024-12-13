from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset = load_iris()

X_train,X_test,y_train,y_test = train_test_split(dataset["data"],dataset["target"])
# print("the training dataset is mentioned below:")
# print(X_train.shape)
# print("the testing dataset is mentioned below:")
# print(X_test.shape)

model = KNeighborsClassifier()
model.fit(X_train,y_train)

res = model.predict(X_test)
print(res)

print(" the output is :",dataset["target_names"][res])

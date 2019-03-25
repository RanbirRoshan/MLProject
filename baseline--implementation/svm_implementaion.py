import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


dataset = pd.read_csv(r'''../music_csv_data/data.csv''')
print(dataset)

X, y = dataset.iloc[:,1:26], dataset.iloc[:,27]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

classifier = svm.SVC(C=1, gamma="scale", kernel="rbf", decision_function_shape="ovo")
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))

print("--")
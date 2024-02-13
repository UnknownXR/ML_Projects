import pandas as pd
from sklearn import preprocessing 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv('Dataset-Mental-Disorders.csv')
print("Missing values in Dataset")
print(df.isnull().sum())

print("Converted Values")
column_mappings = {}
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoder= preprocessing.LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
    column_mappings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(df.head())

for column, mapping in column_mappings.items():
    print(f"{column}: {mapping}")

X = df.iloc[:, 1:-1]
#print(X)
Y = df.iloc[:,-1]
#print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

new_example = [[3,3,2,2,0,0,0,0,0,0,0,1,1,1,4,4,5]]  
prediction = clf.predict(new_example)
print(prediction)
if prediction[0] == 0:
    print("Bipolar Type-1")
elif prediction == 1:
    print("Bipolar Type-2")
elif prediction == 2:
    print("Depression")
else:
    print("Normal")


plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=X.columns, class_names=label_encoder.classes_, filled=True, rounded=True)
plt.show()

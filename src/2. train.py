import sys, os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


outDirTotal = "features_minimal"
fileTotal = os.path.join(outDirTotal, "Features")+".pickle"
A, B = [], []

#LoadFeatures
print("")
print("Reading Features from "+ fileTotal)
pickleF = open(fileTotal, 'rb')
A.extend(pickle.load(pickleF))
B.extend(pickle.load(pickleF))
pickleF.close()

# train model
X_train, X_test, y_train, y_test = train_test_split(A, B, test_size=0.2, random_state=0)
model = MLPClassifier(solver='sgd',alpha=1e-5,hidden_layer_sizes=(250,200,150),
                    batch_size=32, activation='relu', random_state=1, learning_rate='adaptive')
#model = (SVC(kernel='linear', C=0.3))
#model = (LogisticRegression(random_state=0, max_iter=100, n_jobs=-1))
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Train Score: "+str(train_score))
print("Test Score: "+str(test_score))

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
print("Save complete!")
# some time later...

#precision
le = LabelEncoder()
labels =['Cat','Dog']
labels = le.fit_transform(labels)
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))

#matrix
pred = model.predict(X_test)
cm = confusion_matrix(pred,y_test)
print(cm)
sns.heatmap(cm, annot=True)
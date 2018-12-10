import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,classification_report,recall_score
from sklearn import metrics
import time as tt

import texttable
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore') #To ignore warnings

#reading data and defining column names
df = pd.read_csv('/home/potti/Desktop/Btp_Project/data_set.csv',
	names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'])
df.head() #display's attribute names and 5 rows from dataframe

rows_before = len(df) #no.of rows in df before removing null values
print('No.of rows in data_set are : ', rows_before)

df = df.dropna(how='any',axis=0) #delete rows with null values

rows_after = len(df)#no.of rows in df after removing null values
print('No.of rows after removing null values in data_set are : ',rows_after)

#store feature matrix and response vector
X_features = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']] # feature matrix
print(X_features.head())
Y_labels = df[['num']]   # response vector
#print(Y_labels.head())

# splitting X and Y into training and testing sets 
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_labels, test_size=0.10,random_state = 50) 
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X_features, Y_labels, test_size=0.20,random_state =50)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X_features, Y_labels, train_size=0.25,random_state = 50)

#print(Y_train,"\n",Y_test)

print("********************************************************************************")
print("                          Naive Bayes Classifier                                ")
print("********************************************************************************")
# fitting the model to the training set
gauss_clf = GaussianNB()
t0 = tt.time()
gauss_clf.fit(X_train, Y_train)
print("Time taken by the training data is(90%) :", round(tt.time()-t0, 8), "s")
d0 = round(tt.time()-t0,4) #time taken for building model(train)

gauss_clf1 = GaussianNB()
t1 = tt.time()
gauss_clf1.fit(X1_train, Y1_train)
print("Time taken by the training data is(80%) :", round(tt.time()-t1, 8), "s")
d1 = round(tt.time()-t1,4)

gauss_clf2 = GaussianNB()
t2 = tt.time()
gauss_clf2.fit(X2_train, Y2_train)
print("Time taken by the training data is(75%) :", round(tt.time()-t2, 8), "s")
d2 = round(tt.time()-t2,4)

#predictions
t3 = tt.time()
pred_gauss = gauss_clf.predict(X_test)
print("Time taken by the testing data is(10%) :", round(tt.time()-t3, 8), "s")
d3 = round(tt.time()-t3,4) #time taken for prediction

t4 = tt.time()
pred_gauss1 = gauss_clf1.predict(X1_test)
print("Time taken by the testing data is(20%) :", round(tt.time()-t4, 8), "s")
d4 = round(tt.time()-t4,4)

t5 = tt.time()
pred_gauss2 = gauss_clf2.predict(X2_test)
print("Time taken by the testing data is(25%) :", round(tt.time()-t5, 8), "s")
d5 = round(tt.time()-t5,4)
print("\n")

#confusion matrix
confusion_matrix = metrics.confusion_matrix(Y_test, pred_gauss)
print("Confusion matrix(testing data = 10%) :\n",confusion_matrix)
print("\n")

confusion_matrix1 = metrics.confusion_matrix(Y1_test, pred_gauss1)
print("Confusion matrix(testing data = 20%) :\n",confusion_matrix1)
print("\n")

confusion_matrix2 = metrics.confusion_matrix(Y2_test, pred_gauss2)
print("Confusion matrix(testing data = 25%) :\n",confusion_matrix2)
print("\n")

#calculate the accuracy
accuracy_gauss = accuracy_score(Y_test, pred_gauss) 
accuracy_gauss1 = accuracy_score(Y1_test, pred_gauss1)
accuracy_gauss2 = accuracy_score(Y2_test, pred_gauss2)

#classification report
print("\nClassification_report(testing data = 10%) :\n" ,metrics.classification_report(Y_test, pred_gauss))
print("\nClassification_report(testing data = 20%) :\n" ,metrics.classification_report(Y1_test, pred_gauss1))
print("\nClassification_report(testing data = 25%) :\n" ,metrics.classification_report(Y2_test, pred_gauss2))

#calculate recall
recall_gauss = recall_score(Y_test, pred_gauss,average="weighted")
recall_gauss1 = recall_score(Y1_test, pred_gauss1,average="weighted")
recall_gauss2 = recall_score(Y2_test, pred_gauss2,average="weighted")

#calculate precision
prec_gauss = precision_score(Y_test, pred_gauss,average="weighted")
prec_gauss1 = precision_score(Y1_test, pred_gauss1,average="weighted")
prec_gauss2 = precision_score(Y2_test, pred_gauss2,average="weighted")
print("********************************************************************************\n")


print("********************************************************************************")
print("                          K Nearest Neighbors                                   ")
print("********************************************************************************")
#training
knn = KNeighborsClassifier(n_neighbors=60)
t6 = tt.time()
knn.fit(X_train, Y_train)
print("Time taken by the training data is(90%) :", round(tt.time()-t6, 8), "s")
d6 = round(tt.time()-t6,4)

knn1 = KNeighborsClassifier(n_neighbors=20)
t7 = tt.time()
knn1.fit(X1_train, Y1_train)
print("Time taken by the training data is(80%) :", round(tt.time()-t7, 8), "s")
d7 = round(tt.time()-t7,4)

knn2 = KNeighborsClassifier(n_neighbors=20)
t8 = tt.time()
knn2.fit(X2_train, Y2_train)
print("Time taken by the training data is(75%) :", round(tt.time()-t8, 8), "s")
d8 = round(tt.time()-t8,4)

#prediction
t9 = tt.time()
pred_knneigh = knn.predict(X_test)
print("Time taken by the testing data is(10%) :", round(tt.time()-t9, 8), "s")
d9 = round(tt.time()-t9,4)

t10 = tt.time()
pred_knneigh1 = knn1.predict(X1_test)
print("Time taken by the testing data is(20%) :", round(tt.time()-t10, 8), "s")
d10 = round(tt.time()-t10,4)

t11 = tt.time()
pred_knneigh2 = knn2.predict(X2_test)
print("Time taken by the testing data is(25%) :", round(tt.time()-t11, 8), "s")
d11 = round(tt.time()-t11,4) 
print("\n")

#confusion matrix
confusion_matrix3 = metrics.confusion_matrix(Y_test, pred_knneigh)
print("Confusion matrix(testing data = 10%) :\n",confusion_matrix3)
print("\n")
confusion_matrix4 = metrics.confusion_matrix(Y1_test, pred_knneigh1)
print("Confusion matrix(testing data = 20%) :\n",confusion_matrix4)
print("\n")
confusion_matrix5 = metrics.confusion_matrix(Y2_test, pred_knneigh2)
print("Confusion matrix(testing data = 25%) :\n",confusion_matrix5)
print("\n")


#accuracy
score_knn = accuracy_score(Y_test, pred_knneigh)
score_knn1 = accuracy_score(Y1_test, pred_knneigh1)
score_knn2 = accuracy_score(Y2_test, pred_knneigh2)

#report
print("\nClassification_report (testing data = 10%):\n" ,metrics.classification_report(Y_test, pred_knneigh))
print("\nClassification_report(testing data = 20%):\n" ,metrics.classification_report(Y1_test, pred_knneigh1))
print("\nClassification_report(testing data = 25%):\n" ,metrics.classification_report(Y2_test, pred_knneigh2))

#recall
recall_knn = recall_score(Y_test, pred_knneigh,average="weighted")
recall_knn1 = recall_score(Y1_test, pred_knneigh1,average="weighted")
recall_knn2 = recall_score(Y2_test, pred_knneigh2,average="weighted")

#precision
prec_knn = precision_score(Y_test, pred_knneigh,average="weighted")
prec_knn1 = precision_score(Y1_test, pred_knneigh1,average="weighted")
prec_knn2 = precision_score(Y2_test, pred_knneigh2,average="weighted")
print("********************************************************************************\n")


print("********************************************************************************")
print("                          Decision Tree Classifier                              ")
print("********************************************************************************")
#testing
tree_clf = DecisionTreeClassifier(min_samples_split=50,random_state=0)
t12=tt.time()
tree_clf.fit(X_train, Y_train)
print("Time taken by the training data is(90%) :", round(tt.time()-t12, 8), "s")
d12 = round(tt.time()-t12, 4)

tree_clf1 = DecisionTreeClassifier(min_samples_split=50,random_state=0)
t13=tt.time()
tree_clf1.fit(X1_train, Y1_train)
print("Time taken by the training data is(80%) :", round(tt.time()-t13, 8), "s")
d13 = round(tt.time()-t13, 4)

tree_clf2 = DecisionTreeClassifier(min_samples_split=40,random_state=0)
t14=tt.time()
tree_clf2.fit(X2_train, Y2_train)
print("Time taken by the training data is(75%) :", round(tt.time()-t14, 8), "s")
d14 = round(tt.time()-t14, 4)

#prediction
t15 = tt.time()
pred_tree = tree_clf.predict(X_test)
print("Time taken by the testing data is(10%) :", round(tt.time()-t15, 8), "s")
d15 = round(tt.time()-t15, 4)

t16 = tt.time()
pred_tree1 = tree_clf1.predict(X1_test)
print("Time taken by the testing data is(20%) :", round(tt.time()-t16, 8), "s")
d16 = round(tt.time()-t16, 4)

t17 = tt.time()
pred_tree2 = tree_clf2.predict(X2_test)
print("Time taken by the testing data is(25%) :", round(tt.time()-t17, 8), "s")
d17 = round(tt.time()-t17, 4)
print("\n")

#matrix
confusion_matrix6 = metrics.confusion_matrix(Y_test, pred_tree)
print("Confusion matrix(testing data = 10%) :\n",confusion_matrix6)
print("\n")
confusion_matrix7 = metrics.confusion_matrix(Y1_test, pred_tree1)
print("Confusion matrix(testing data = 20%) :\n",confusion_matrix7)
print("\n")
confusion_matrix8 = metrics.confusion_matrix(Y2_test, pred_tree2)
print("Confusion matrix(testing data = 25%) :\n",confusion_matrix8)
print("\n")

#accuracy
score_tree = accuracy_score(Y_test, pred_tree)
score_tree1 = accuracy_score(Y1_test, pred_tree1)
score_tree2 = accuracy_score(Y2_test, pred_tree2)

#report
print("\nClassification_report (testing data = 10%):\n" ,metrics.classification_report(Y_test, pred_tree))
print("\nClassification_report(testing data = 20%):\n" ,metrics.classification_report(Y1_test, pred_tree1))
print("\nClassification_report(testing data = 25%):\n" ,metrics.classification_report(Y2_test, pred_tree2))

#recall
recall_tree = recall_score(Y_test, pred_tree,average="weighted")
recall_tree1 = recall_score(Y1_test, pred_tree1,average="weighted")
recall_tree2 = recall_score(Y2_test, pred_tree2,average="weighted")

#precision
prec_tree = precision_score(Y_test, pred_tree,average="weighted")
prec_tree1 = precision_score(Y1_test, pred_tree1,average="weighted")
prec_tree2 = precision_score(Y2_test, pred_tree2,average="weighted")

print("********************************************************************************\n")


print("********************************************************************************")
print("                          Random Forest Classifier                              ")
print("********************************************************************************")
#training
randfor_clf = RandomForestClassifier(min_samples_split=50,random_state=60)
t18=tt.time()
randfor_clf.fit(X_train, Y_train)
print("Time taken by the training data is(90%) :", round(tt.time()-t18, 8), "s")
d18 = round(tt.time()-t18, 5)

randfor_clf1 = RandomForestClassifier(min_samples_split=30,random_state=0)
t19=tt.time()
randfor_clf1.fit(X1_train, Y1_train)
print("Time taken by the training data is(80%) :", round(tt.time()-t19, 8), "s")
d19 = round(tt.time()-t19, 5)

randfor_clf2 = RandomForestClassifier(min_samples_split=10,random_state=0)
t20=tt.time()
randfor_clf2.fit(X2_train, Y2_train)
print("Time taken by the training data is(75%) :", round(tt.time()-t20, 8), "s")
d20 = round(tt.time()-t20, 5)

#prediction
t21=tt.time()
pred_randfor = randfor_clf.predict(X_test)
print("Time taken by the testing data is(10%) :", round(tt.time()-t21, 8), "s")
d21 = round(tt.time()-t21, 5)

t22=tt.time()
pred_randfor1 = randfor_clf1.predict(X1_test)
print("Time taken by the testing data is(20%) :", round(tt.time()-t22, 8), "s")
d22 = round(tt.time()-t22, 5)

t23=tt.time()
pred_randfor2 = randfor_clf2.predict(X2_test)
print("Time taken by the testing data is(25%) :", round(tt.time()-t23, 8), "s")
d23 = round(tt.time()-t23, 5)
print("\n")

#matrix
confusion_matrix9 = metrics.confusion_matrix(Y_test, pred_randfor)
print("Confusion matrix(testing data = 10%) :\n",confusion_matrix9)
print("\n")
confusion_matrix10 = metrics.confusion_matrix(Y1_test, pred_randfor1)
print("Confusion matrix(testing data = 20%) :\n",confusion_matrix10)
print("\n")
confusion_matrix11 = metrics.confusion_matrix(Y2_test, pred_randfor2)
print("Confusion matrix(testing data = 25%) :\n",confusion_matrix11)

#accuracy
score_randfor = accuracy_score(Y_test, pred_randfor)
score_randfor1 = accuracy_score(Y1_test, pred_randfor1)
score_randfor2 = accuracy_score(Y2_test, pred_randfor2)

#report
print("\nClassification_report (testing data = 10%):\n" ,metrics.classification_report(Y_test, pred_tree))
print("\nClassification_report(testing data = 20%):\n" ,metrics.classification_report(Y1_test, pred_tree1))
print("\nClassification_report(testing data = 25%):\n" ,metrics.classification_report(Y2_test, pred_tree2))

#recall
recall_randfor = recall_score(Y_test, pred_randfor,average="weighted")
recall_randfor1 = recall_score(Y1_test, pred_randfor1,average="weighted")
recall_randfor2 = recall_score(Y2_test, pred_randfor2,average="weighted")

#precision
prec_randfor = precision_score(Y_test, pred_randfor,average="weighted")
prec_randfor1 = precision_score(Y1_test, pred_randfor1,average="weighted")
prec_randfor2 = precision_score(Y2_test, pred_randfor2,average="weighted")
print("********************************************************************************\n")

#------------------------------90%--------------------------------------#
table1 = texttable.Texttable()
table1.add_rows([["Classifier", "Accuracy","Precision","Recall"], 
	["Naive Bayes",accuracy_gauss,prec_gauss,recall_gauss], 
	["KNN",score_knn,prec_knn,recall_knn], ["Decision Tree",score_tree,prec_tree,recall_tree],
	["Random Forest",score_randfor,prec_randfor,recall_randfor]])
print("When training data is 90%\n")
print(table1.draw())

#------------------------------80%--------------------------------------#
table2 = texttable.Texttable()
table2.add_rows([["Classifier", "Accuracy","Precision","Recall"], 
	["Naive Bayes",accuracy_gauss1,prec_gauss1,recall_gauss1], 
	["KNN",score_knn1,prec_knn1,recall_knn1], ["Decision Tree",score_tree1,prec_tree1,recall_tree1],
	["Random Forest",score_randfor1,prec_randfor1,recall_randfor1]])
print("When training data is 80%\n")
print(table2.draw())

#------------------------------75%--------------------------------------#
table3 = texttable.Texttable()
table3.add_rows([["Classifier", "Accuracy","Precision","Recall"], 
	["Naive Bayes",accuracy_gauss2,prec_gauss2,recall_gauss2], 
	["KNN",score_knn2,prec_knn2,recall_knn2], ["Decision Tree",score_tree2,prec_tree2,recall_tree2],
	["Random Forest",score_randfor2,prec_randfor2,recall_randfor2]])
print("When training data is 75%\n")
print(table3.draw())

#print(gauss_clf.predict([[57,1,3,150,168,0,0,174,0,1.6,1,0,3]]))
#print(gauss_clf.predict([[60,1,4,140,293,0,2,170,0,1.2,2,2,7]]))


########90%###########
n_groups = 4

a1 = accuracy_gauss;a2 = score_knn;a3=score_tree;a4 = score_randfor
b1=prec_gauss;b2 = prec_knn;b3 = prec_tree;b4=prec_randfor
c1 = recall_knn;c2 = recall_knn;c3=recall_tree;c4=recall_randfor

acc = (a1, a2, a3, a4)
prec = (b1, b2, b3, b4)
rec =(c1,c2,c3,c4)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8
 
rects1 = plt.bar(index, acc, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy')
 
rects2 = plt.bar(index + bar_width, prec, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Precision')
 
rects3 = plt.bar(index + bar_width+bar_width, rec, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Recall')
 
plt.xlabel('Method')
plt.ylabel('Scores')
plt.title('Scores by Methods for training 90%')
plt.xticks(index + bar_width, ('Naive Bayes', 'KNN', 'Decision Tree', 'Random Forest'))
plt.legend()
 
plt.tight_layout()
plt.show()

####### 2nd grph(80%) ##########

g1 = accuracy_gauss1;g2 = score_knn1;g3=score_tree1;g4 = score_randfor1
e1=prec_gauss1;e2 = prec_knn1;e3 = prec_tree1;e4=prec_randfor1
f1 = recall_knn1;f2 = recall_knn1;f3=recall_tree1;f4=recall_randfor1

acc1 = (g1, g2, g3, g4)
prec1 = (e1, e2, e3, e4)
rec1 = (f1,f2,f3,f4)
 
rects1 = plt.bar(index, acc1, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy')
 
rects2 = plt.bar(index + bar_width, prec1, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Precision')
 
rects3 = plt.bar(index + bar_width+bar_width, rec1, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Recall')
 
plt.xlabel('Method')
plt.ylabel('Scores')
plt.title('Scores by Methods for training 80%')
plt.xticks(index + bar_width, ('Naive Bayes', 'KNN', 'Decision Tree', 'Random Forest'))
plt.legend()
 
plt.tight_layout()
plt.show()

############ 3rd grap(75%) ##############
h1 = accuracy_gauss2;h2 = score_knn2;h3=score_tree2;h4 = score_randfor2
i1=prec_gauss2;i2 = prec_knn2;i3 = prec_tree2;i4=prec_randfor2
j1 = recall_knn2;j2 = recall_knn2;j3=recall_tree2;j4=recall_randfor2

acc2 = (h1, h2, h3, h4)
prec2 = (i1, i2, i3, i4)
rec2 = (j1,j2,j3,j4)
 
rects1 = plt.bar(index, acc2, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy')
 
rects2 = plt.bar(index + bar_width, prec2, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Precision')
 
rects3 = plt.bar(index + bar_width+bar_width, rec2, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Recall')
 
plt.xlabel('Method')
plt.ylabel('Scores')
plt.title('Scores by Methods for training 75%')
plt.xticks(index + bar_width, ('Naive Bayes', 'KNN', 'Decision Tree', 'Random Forest'))
plt.legend()
plt.tight_layout()
plt.show()


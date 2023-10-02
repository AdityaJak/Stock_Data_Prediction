#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 18:41:36 2022

@author: adithya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from numpy.polynomial.polynomial import polyfit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import GaussianNB
from scipy.stats import t


pd.options.mode.chained_assignment = None  # default='warn'

df_stock = pd.read_csv("TSLA.csv")

df_stock.set_index(['Date'])

df_stock_latest = df_stock[(df_stock['Date'] >= "2022-01-29") & (df_stock['Date'] <= "2023-01-29")]
df_weekly = df_stock_latest.groupby(['Week_Number'])

def get_features_with_labels():
    model_df = pd.DataFrame()
    mean_list = []
    std_list = []
    labels = []
    for current_group in df_weekly:
        for index, cur_tuple in enumerate(current_group):
            if index ==0:
                continue
            mean = cur_tuple['Return'].mean()
            std = cur_tuple['Return'].std()
            isIncrease = pd.Series(cur_tuple['Return'].tail(2)).is_monotonic_increasing
            label = 0 # red
            if isIncrease:
                label = 1 # green
            mean_list.append(mean)
            std_list.append(std)

            labels.append(label)
    #break
    model_df['mean'] = mean_list
    model_df['std'] = std_list
    model_df['label'] = labels
    return model_df

model_df = get_features_with_labels()

print(model_df)


fig = sns.scatterplot(x='mean', y='std', hue='label', data=model_df)
plt.show()
fig = sns.scatterplot(x='mean', y='std', hue='label', data=model_df)
x = [-0.02, 0.025]
y = [0, 0.035]

plt.plot(x, y, linewidth=3, color='green')
plt.show()


# Fit to a logistic regression model
XTrain = model_df.loc[:, model_df.columns != 'label']
YTrain = model_df.loc[:, model_df.columns == 'label']

XTrain = np.asarray(XTrain)
YTrain = np.asarray(YTrain).squeeze()

model = LogisticRegression(random_state=42)
model.fit(XTrain, YTrain)
print("lr model score: ", round(model.score(XTrain, YTrain), 2))

preds = model.predict(XTrain)

print("preds: ", preds)
#The accuracy of the logistic regression model is 52%

print("The confusion matrix is",confusion_matrix(YTrain, preds))
cn_mtrx = confusion_matrix(YTrain, preds)

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)
#The true positive rate is 1.0 and the true negative rate is 0.0 with an accuracy of 52%

#Fit to knn model
no_neighbors = [3, 5, 7, 9, 11]
train_accuracy = np.empty(len(no_neighbors))

for i, k in enumerate(no_neighbors):
    # Instantiate the classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to training data
    knn.fit(XTrain, YTrain)

train_accuracy[i] = knn.score(XTrain, YTrain)
print("Train Accuracy is ",train_accuracy)
print("knn model score: ", round(knn.score(XTrain, YTrain), 2))

preds = knn.predict(XTrain)

print("preds: ", preds)

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(no_neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#Optimal value of k
error_rate = []
for i in range(1,11):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(XTrain,YTrain)
 pred_i = knn.predict(XTrain)
 error_rate.append(np.mean(pred_i != preds))

plt.figure(figsize=(10,6))
plt.plot(range(1,11),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
#optimal value of k = 6 at a minimum error of 15%
#Accuracy of the model is 62%

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(XTrain, YTrain)
print("knn model score: ", round(knn.score(XTrain, YTrain), 2))
preds = knn.predict(XTrain)
print("preds: ", preds)
#The accuracy using the optimal value of k is 62%
print("The confusion matrix is",confusion_matrix(pred_i, preds))
cn_mtrx = confusion_matrix(pred_i, preds)

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)
#The true positive rate is 0.87 and the true negative rate is 0.47 with an accuracy of 73%

# Linear regression
W = [5,6,7,8,9,10,11,12]
poly1 = LinearRegression(n_jobs=-4)
poly1.fit(XTrain, YTrain)
print("poly1 model score: ", round(poly1.score(XTrain, YTrain), 2))
pred1 = poly1.predict(XTrain)

print("preds: ", pred1)
print("The confusion matrix is",confusion_matrix(YTrain, preds))
cn_mtrx = confusion_matrix(YTrain, preds)
train_accuracy = poly1.score(XTrain, YTrain)
print("Train Accuracy is ",train_accuracy)
plt.title('Linear Regression')
plt.plot(pred1, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Weeks')
plt.ylabel('Accuracy')
plt.show()

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)

# Quadratic Regression
poly2 = make_pipeline(PolynomialFeatures(2), Ridge())
poly2.fit(XTrain, YTrain)
print("poly2 model score: ", round(poly2.score(XTrain, YTrain), 2))
pred2 = poly2.predict(XTrain)

print("preds: ", pred2)
train_accuracy = poly2.score(XTrain, YTrain)
print("Train Accuracy is ",train_accuracy)
plt.title('Quadratic Regression')
plt.plot(pred2, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Weeks')
plt.ylabel('Accuracy')
plt.show()

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)

# Quadratic Regression 2
poly3 = make_pipeline(PolynomialFeatures(3), Ridge())
poly3.fit(XTrain, YTrain)
print("poly3 model score: ", round(poly3.score(XTrain, YTrain), 2))
pred3 = poly3.predict(XTrain)

print("preds: ", pred3)
train_accuracy = poly3.score(XTrain, YTrain)
print("Train Accuracy is ",train_accuracy)
plt.title('Quadratic Regression 2')
plt.plot(pred3, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Weeks')
plt.ylabel('Accuracy')
plt.show()

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)

# Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(XTrain, YTrain)
print("Naive Bayes model score: ", round(nb.score(XTrain, YTrain), 2))
pred4 = nb.predict(XTrain)

print("preds: ", pred4)
#The accuracy using the Gaussian Naive Bayes model is 63%
train_accuracy = nb.score(XTrain, YTrain)
print("Train Accuracy is ", train_accuracy)
plt.title('Gaussian Naive Bayes')
plt.plot(pred4, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Weeks')
plt.ylabel('Accuracy')
plt.show()

print("The confusion matrix is",confusion_matrix(pred4, preds))
cn_mtrx = confusion_matrix(pred4, preds)

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)

#Using decision tree classifier with criterion as entropy
dt_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5)
dt_entropy.fit(XTrain, YTrain)

print("DTree model score: ", round(dt_entropy.score(XTrain, YTrain), 2))
pred5 = dt_entropy.predict(XTrain)
print("preds: ", pred5)
#The accuracy using the DTree model is 71%
plt.title('Decision Tree')
plt.plot(pred5, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Weeks')
plt.ylabel('Accuracy')
plt.show()
print("The confusion matrix is",confusion_matrix(pred5, preds))
cn_mtrx = confusion_matrix(pred5, preds)

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)

#Using Random Forest classifier with criterion as entropy
# Number of trees in random forest
n_estimators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Maximum number of levels in tree
max_depth = [1, 2, 3, 4, 5]
accuracy = []
for i in range(1,11):
    for j in range(1,6):
        rf = RandomForestClassifier(criterion= "entropy", n_estimators = i, random_state=1, max_depth = j)
        rf.fit(XTrain, YTrain)
        print("Random Forest model score: ", round(rf.score(XTrain, YTrain), 2))
        pred6 = rf.predict(XTrain)
        print("preds: ", pred6)
        rf_error_rate = np.mean(pred6 != YTrain)
        rf_Accuracy = 1 - rf_error_rate
        accuracy.append([i,j,round(rf_Accuracy*100,3),pred6])
max_list = max(accuracy, key=lambda sublist: sublist[2])
prediction = list(max_list[3])
print("The best combination is:", max_list[0], max_list[1])
print("The accuracy of the best combination of N and d is", max_list[2])
N_values = []
D_values = []
Accuracy_values = []
for k in range(len(accuracy)):
    N_values.append(accuracy[k][0])
    D_values.append(accuracy[k][1])
    Accuracy_values.append(accuracy[k][2])
print(Accuracy_values)
#The accuracy using the Random Forest classifier model for best N and d is 94%
plt.title('Random Forest')
plt.plot(pred6, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Weeks')
plt.ylabel('Accuracy')
plt.show()
print("The confusion matrix is",confusion_matrix(pred6, preds))
cn_mtrx = confusion_matrix(pred6, preds)

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)

#Using Linear SVM classifier
svm = SVC(kernel='linear')
svm.fit(XTrain, YTrain)
print("Linear SVM model score: ", round(svm.score(XTrain, YTrain), 2))
pred7 = svm.predict(XTrain)
print("preds: ", pred7)
#The accuracy using the Linear SVM classifier model is 52%
plt.title('Linear SVM')
plt.plot(pred6, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Weeks')
plt.ylabel('Accuracy')
plt.show()
print("The confusion matrix is",confusion_matrix(pred7, preds))
cn_mtrx = confusion_matrix(pred7, preds)

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)

#Using Gaussian SVM classifier
svmg = SVC(kernel='rbf')
svmg.fit(XTrain, YTrain)
print("Gaussian SVM model score: ", round(svmg.score(XTrain, YTrain), 2))
pred8 = svmg.predict(XTrain)
print("preds: ", pred8)
#The accuracy using the Gaussian SVM classifier model is 63%
#The Gaussian SVM is better than the Linear SVM
plt.title('Gaussian SVM')
plt.plot(pred6, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Weeks')
plt.ylabel('Accuracy')
plt.show()
print("The confusion matrix is",confusion_matrix(pred8, preds))
cn_mtrx = confusion_matrix(pred8, preds)

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)

#Using Polynomial SVM classifier
svmp = SVC(kernel='poly', degree=2)
svmp.fit(XTrain, YTrain)
print("Polynomial SVM model score: ", round(svmp.score(XTrain, YTrain), 2))
pred9 = svmp.predict(XTrain)
print("preds: ", pred9)
#The accuracy using the Polynomial SVM classifier model is 60%
#The Polynomial SVM with degree 2 is better than the Linear SVM
plt.title('Polynomial SVM')
plt.plot(pred6, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Weeks')
plt.ylabel('Accuracy')
plt.show()
print("The confusion matrix is",confusion_matrix(pred9, preds))
cn_mtrx = confusion_matrix(pred9, preds)

FP = cn_mtrx.sum(axis=0) - np.diag(cn_mtrx)
FN = cn_mtrx.sum(axis=1) - np.diag(cn_mtrx)
TP = np.diag(cn_mtrx)
TN = cn_mtrx.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

TPR = TP/(TP+FN)
print("The true positive rate is",TPR)
TNR = TN/(TN+FP)
print("The true negative rate is",TNR)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("The Accuracy is", ACC)

#Using K-means clustering with k = 3
inertia = []
K = range(1,11)
for k in K:
    km = KMeans(n_clusters=3, init='k-means++', random_state=0)
    km = km.fit(model_df)
    inertia.append(km.inertia_)
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method For k = 3')
plt.show()

#Using knee method to find best k
def find_k(increment=0, decrement=0):
    distortion = {}
    for k in range(1, 8):
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(model_df)
        distortion[k] = kmeans.inertia_
    kn = KneeLocator(x=list(distortion.keys()), y=list(distortion.values()), curve='convex', direction='increasing')
    k = kn.knee + increment - decrement
    return k
print('Optimal value of k using knee method is', k)

kmeans = KMeans(n_clusters=10, random_state=100)
kmeans.fit(model_df)
y_kmeans = kmeans.predict(model_df)
print(y_kmeans)
centers = np.array(kmeans.cluster_centers_)
print(centers)
centroid = pd.DataFrame(centers)
print(centroid)
sns.scatterplot(x = centers[:,0], y = centers[:,1], marker="o", color='r', s = 70, label="centroid")

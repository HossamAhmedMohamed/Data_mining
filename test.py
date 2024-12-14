import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# قراءة البيانات
X = pd.read_csv('RTA Dataset (2).csv')
X
X.info()
X.isnull()
X.isnull().sum()

# التعامل مع القيم المفقودة
imputer = SimpleImputer(strategy="most_frequent")
X[['Vehicle_driver_relation']] = imputer.fit_transform(X[['Vehicle_driver_relation']])
X
imputer = SimpleImputer(strategy="most_frequent")
X[['Type_of_collision']] = imputer.fit_transform(X[['Type_of_collision']])
X
imputer = SimpleImputer(strategy="most_frequent")
X[['Types_of_Junction']] = imputer.fit_transform(X[['Types_of_Junction']])
X
imputer = SimpleImputer(strategy="most_frequent")
X[['Vehicle_movement']] = imputer.fit_transform(X[['Vehicle_movement']])
X
X.isnull().sum()

# التعامل مع أنواع البيانات
X.Pedestrian_movement.apply(type).unique()
X.Accident_severity.apply(type).value_counts()

# إيجاد مواقع القيم غير الصحيحة
locations = X.index[X.Accident_severity.apply(type) == int].tolist()
locations
X[['Number_of_vehicles_involved','Number_of_casualties']].describe().round(2)

# رسم boxplot
fig, axs = plt.subplots(2,1,dpi=90, figsize=(5,10))
i = 0
for col in X[['Number_of_vehicles_involved','Number_of_casualties']]:
    axs[i].boxplot(X[[col]], vert=False)
    axs[i].set_ylabel(col)
    i += 1
plt.show()

# حساب Q1 و Q3 و IQR
Q1 = X.Number_of_casualties.quantile(0.25)
Q3 = X.Number_of_casualties.quantile(0.75)
IQR = Q3 - Q1

# تحديد الحدود القصوى والدنيا
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# العثور على القيم الشاذة
outliers = X[(X.Number_of_casualties < lower_bound) | (X.Number_of_casualties  > upper_bound)]

# عرض القيم الشاذة
print("Outliers:")
print(outliers)

# التعامل مع القيم الشاذة لعمود "Number_of_vehicles_involved"
Q1 = X.Number_of_vehicles_involved.quantile(0.25)
Q3 = X.Number_of_vehicles_involved.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = X[(X.Number_of_vehicles_involved < lower_bound) | (X.Number_of_vehicles_involved  > upper_bound)]
print("Outliers:")
print(outliers)

# تطبيق MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X[['Number_of_vehicles_involved']] = scaler.fit_transform(X[['Number_of_vehicles_involved']])
print(X['Number_of_vehicles_involved'])

# تطبيق MinMaxScaler لعمود "Number_of_casualties"
scaler = MinMaxScaler(feature_range=(0,1))
X[['Number_of_casualties']] = scaler.fit_transform(X[['Number_of_casualties']])
print(X['Number_of_casualties'])

# حساب توزيع القيم
value_counts = X['Sex_of_driver'].value_counts()
print(value_counts)

# رسم توزيع البيانات
plt.figure(figsize=(8, 5))
value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Accident Type Distribution")
plt.xlabel("Accident Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# العثور على التكرارات
duplicates = X[X.duplicated()]
print(duplicates)

# حفظ البيانات في ملف CSV
X.to_csv('cleandData2.csv', index=False)

# دوال التحميل والتقسيم
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def determine_features_and_goal(X):
    features = X.drop(columns=['Sex_of_driver'])  # العمود الذي يحتوي على الهدف
    goal = X['Sex_of_driver']
    return features, goal

# ترميز البيانات الفئوية
def label_encode_categorical_features(X):
    le = LabelEncoder()
    X = X.apply(le.fit_transform)
    return X

# تقسيم البيانات للتدريب والاختبار
def split_data(features, goal, test_size=0.5, random_state=3):
    X_train, X_test, y_train, y_test = train_test_split(features, goal, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# تطبيق الخوارزميات
def apply_knn_classifier(K, X_train, X_test, y_train):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

# خوارزمية Naive Bayes
def apply_naive_bayes_classifier(X_train, X_test, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb.predict(X_test)

# خوارزمية Decision Tree
def apply_decision_tree_classifier(X_train, X_test, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    return dt.predict(X_test), dt

# خوارزمية Random Forest
from sklearn.ensemble import RandomForestClassifier
def apply_random_forest_classifier(X_train, X_test, y_train):
    rm = RandomForestClassifier(n_estimators=10, max_depth=25, criterion="gini", min_samples_split=10)
    rm.fit(X_train, y_train)
    rm_prd = rm.predict(X_test)
    return rm, rm_prd

# حساب الأداء
def calculate_performance(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    v = round(metrics.accuracy_score(y_test, y_pred) * 100)
    w = round(metrics.precision_score(y_test, y_pred, average='macro') * 100)
    z = round(metrics.recall_score(y_test, y_pred, average='macro') * 100)
    return v, w, z, cm

if __name__ == '__main__':
    file_path = 'cleandData2.csv'
    X = load_csv(file_path)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    x, y = X.shape
    features, goal = determine_features_and_goal(X)
    target = goal.tolist()
    target = list(set(goal))  # الحصول على القيم الفريدة
    print(target)
    print(len(target))

    # حفظ أسماء الميزات
    F = list(features)
    print(F)

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = split_data(features, goal, test_size=0.3, random_state=3)
    X_train = label_encode_categorical_features(X_train)
    X_test = label_encode_categorical_features(X_test)

    print(len(X_train))
    print(len(X_test))

    # تطبيق الخوارزميات
    K = round(np.sqrt(x))
    y_pred_knn = apply_knn_classifier(K, X_train, X_test, y_train)
    A_res, P_res, R_res, con = calculate_performance(y_test, y_pred_knn)
    print("KNN report\n", classification_report(y_test, y_pred_knn, target_names=target))
    print("\n KNN Accuracy:", A_res, '%')
    print("KNN Precision:", P_res, '%')
    print("KNN Recall:", R_res, '%')
    print("KNN Confusion Matrix:\n", con)

    y_pred_nb = apply_naive_bayes_classifier(X_train, X_test, y_train)
    A_res, P_res, R_res, con = calculate_performance(y_test, y_pred_nb)
    print("\n Naive Bayes report\n", classification_report(y_test, y_pred_nb, target_names=target))
    print("\n Naive Bayes Accuracy:", A_res, '%')
    print("Naive Bayes Precision:", P_res, '%')
    print("Naive Bayes Recall:", R_res, '%')
    print("Naive Bayes Confusion Matrix:\n", con)

    y_pred_dt, dt_model = apply_decision_tree_classifier(X_train, X_test, y_train)
    A_res, P_res, R_res, con = calculate_performance(y_test, y_pred_dt)
    print("\n Decision Tree report\n", classification_report(y_test, y_pred_dt, target_names=target))
    print("\n Decision Tree Accuracy:", A_res, '%')
    print("Decision Tree Precision:", P_res, '%')
    print("Decision Tree Recall:", R_res, '%')
    print("Decision Tree Confusion Matrix:\n", con)
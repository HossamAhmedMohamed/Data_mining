import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import metrics  # إضافة هذه السطر

# قراءة البيانات
file_path = './data_set_2/accident_data_improved.csv'  # استخدم المسار الصحيح للملف
X = pd.read_csv(file_path)

# التحقق من القيم المفقودة
print("Data before cleaning:")
print(X.info())

# التعامل مع القيم المفقودة
imputer = SimpleImputer(strategy="most_frequent")
X[['Accident_Type']] = imputer.fit_transform(X[['Accident_Type']])
X[['Latitude', 'Longitude']] = imputer.fit_transform(X[['Latitude', 'Longitude']])
X[['Year', 'Month', 'Day', 'Hour', 'Minute']] = imputer.fit_transform(X[['Year', 'Month', 'Day', 'Hour', 'Minute']])

# تحويل الأعمدة 'Location' و 'Accident_Type' إلى قيم عددية باستخدام LabelEncoder
label_encoder_location = LabelEncoder()
X['Location'] = label_encoder_location.fit_transform(X['Location'])

label_encoder_accident_type = LabelEncoder()
X['Accident_Type'] = label_encoder_accident_type.fit_transform(X['Accident_Type'])

# التحقق من البيانات بعد التنظيف
print("Data after cleaning:")
print(X.info())

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['Accident_Type']), X['Accident_Type'], test_size=0.3, random_state=3)

# تطبيق StandardScaler على الأعمدة العددية فقط (Latitude و Longitude)
scaler = StandardScaler()
X_train[['Latitude', 'Longitude']] = scaler.fit_transform(X_train[['Latitude', 'Longitude']])
X_test[['Latitude', 'Longitude']] = scaler.transform(X_test[['Latitude', 'Longitude']])

# تطبيق KNN Classifier
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

# تنفيذ الخوارزميات
if __name__ == '__main__':
    # تطبيق الخوارزميات
    K = round(np.sqrt(len(X)))  # تحديد قيمة K لـ KNN
    y_pred_knn = apply_knn_classifier(K, X_train, X_test, y_train)
    A_res, P_res, R_res, con = calculate_performance(y_test, y_pred_knn)
    print("\nKNN report\n", classification_report(y_test, y_pred_knn))
    print("\nKNN Accuracy:", A_res, '%')
    print("KNN Precision:", P_res, '%')
    print("KNN Recall:", R_res, '%')
    print("KNN Confusion Matrix:\n", con)

    y_pred_nb = apply_naive_bayes_classifier(X_train, X_test, y_train)
    A_res, P_res, R_res, con = calculate_performance(y_test, y_pred_nb)
    print("\nNaive Bayes report\n", classification_report(y_test, y_pred_nb))
    print("\nNaive Bayes Accuracy:", A_res, '%')
    print("Naive Bayes Precision:", P_res, '%')
    print("Naive Bayes Recall:", R_res, '%')
    print("Naive Bayes Confusion Matrix:\n", con)

    y_pred_dt, dt_model = apply_decision_tree_classifier(X_train, X_test, y_train)
    A_res, P_res, R_res, con = calculate_performance(y_test, y_pred_dt)
    print("\nDecision Tree report\n", classification_report(y_test, y_pred_dt))
    print("\nDecision Tree Accuracy:", A_res, '%')
    print("Decision Tree Precision:", P_res, '%')
    print("Decision Tree Recall:", R_res, '%')
    print("Decision Tree Confusion Matrix:\n", con)

    # تطبيق Random Forest
    model_rf, y_pred_rf = apply_random_forest_classifier(X_train, X_test, y_train)
    A_res, P_res, R_res, con = calculate_performance(y_test, y_pred_rf)
    print("\nRandom Forest report\n", classification_report(y_test, y_pred_rf))
    print("\nRandom Forest Accuracy:", A_res, '%')
    print("Random Forest Precision:", P_res, '%')
    print("Random Forest Recall:", R_res, '%')
    print("Random Forest Confusion Matrix:\n", con)

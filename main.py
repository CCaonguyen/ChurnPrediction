import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Bước 1: Đọc và Khám phá dữ liệu
data = pd.read_csv('churn-bigml-80.csv')

# Xem vài dòng đầu của dữ liệu
print(data.head())

# Kiểm tra thống kê cơ bản
print(data.describe())

# Kiểm tra giá trị thiếu
print(data.isnull().sum())

# Vẽ biểu đồ phân phối của biến mục tiêu (Churn)
sns.countplot(data['Churn'])
plt.show()

# Bước 2: Tiền xử lý dữ liệu
data.fillna(0, inplace=True)

# Bước 3: Mã hóa biến chuỗi
data = pd.get_dummies(data, columns=['State', 'International plan', 'Voice mail plan'], drop_first=True)

# Bước 4: Tách dữ liệu thành các tập dữ liệu con
X = data.drop('Churn', axis=1)
y = data['Churn']

# Bước 5: Chuẩn hóa dữ liệu (chỉ chuẩn hóa các đặc trưng số)
numeric_features = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 6: Huấn luyện mô hình
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Bước 7: Đánh giá mô hình
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

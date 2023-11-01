import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Định nghĩa một lớp (class) ChurnPrediction để thực hiện dự đoán việc khách hàng "churn".
class ChurnPrediction:
    def __init__(self, data_path):
# Khởi tạo đối tượng ChurnPrediction với đường dẫn tới tệp dữ liệu.
        self.data = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_classifier = RandomForestClassifier()
#Binary Classification để dự đoán khả năng "churn" của khách hàng (khách hàng ra đi hoặc không ra đi) bằng mô hình RandomForestClassifier.
    def explore_data(self):
# Hiển thị dữ liệu và tính toán các thống kê cơ bản.
        print(self.data.head())
# Hiển thị 5 dòng đầu tiên của dữ liệu.
        print(self.data.describe())
# Tính toán thống kê cơ bản về dữ liệu.
        print(self.data.isnull().sum())
# Kiểm tra giá trị thiếu trong dữ liệu.
        sns.countplot(self.data['Churn'])
        plt.show()
# Vẽ biểu đồ phân phối của biến mục tiêu "Churn".
    def preprocess_data(self):
# Tiền xử lý dữ liệu bằng cách điền giá trị thiếu và mã hóa biến chuỗi.
        self.data.fillna(0, inplace=True)  # Điền giá trị thiếu bằng 0.
        self.data = pd.get_dummies(self.data, columns=['State', 'International plan', 'Voice mail plan'], drop_first=True)
# Mã hóa các biến chuỗi (categorical variables) bằng phương pháp mã hóa one-hot.
    def split_data(self, test_size=0.2, random_state=42):
# Tách dữ liệu thành tập huấn luyện và tập kiểm tra.
        X = self.data.drop('Churn', axis=1)
# Tách biến mục tiêu.
        y = self.data['Churn']
        numeric_features = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)



    def train_model(self):
        self.rf_classifier.fit(self.X_train, self.y_train)
# train_model sử dụng RandomForestClassifier để huấn luyện mô hình phân loại nhị phân (Binary Classification).
#tôi sử dụng mô hình RandomForestClassifier (một mô hình học máy phân loại) để huấn luyện mô hình trên dữ liệu huấn luyện.
#Việc huấn luyện mô hình là quá trình mà mô hình học từ dữ liệu để hiểu các mẫu và mối quan hệ trong dữ liệu.
# Sau quá trình huấn luyện, mô hình có khả năng dự đoán một kết quả cho các dữ liệu mới mà nó chưa thấy trước đó.

    def evaluate_model(self):
        # Đánh giá mô hình và in ra các độ đo đánh giá.
        y_pred = self.rf_classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
# Tỉ lệ phần trăm dự đoán chính xác.
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)
# Ma trận nhầm lẫn.
        class_report = classification_report(self.y_test, y_pred)
        print("Classification Report:\n", class_report)
 # Báo cáo phân loại.
#evaluate_model tính toán các độ đo đánh giá mô hình như accuracy, confusion matrix và classification report để đánh giá khả năng dự đoán của mô hình.
#Hàm này đánh giá khả năng dự đoán của mô hình sau khi đã được huấn luyện.
#y_pred là kết quả dự đoán của mô hình trên tập dữ liệu kiểm tra (self.X_test).
#accuracy là tỷ lệ phần trăm số lượng dự đoán chính xác trên tổng số dự đoán. Nó cung cấp thông tin về mức độ chính xác của mô hình.
#conf_matrix là ma trận nhầm lẫn (confusion matrix) chứa thông tin về số lượng các dự đoán đúng và sai cho từng lớp. Nó giúp bạn hiểu sâu hơn về hiệu suất của mô hình.
#class_report là báo cáo phân loại cung cấp thông tin chi tiết về độ chính xác, recall, F1-score và hỗ trợ cho từng lớp. Nó giúp bạn đánh giá hiệu suất dự đoán cho từng lớp.
# Định nghĩa hàm process_data để thực hiện quá trình khám phá, tiền xử lý, tách dữ liệu, huấn luyện và đánh giá mô hình.
def process_data(churn_predictor):
    churn_predictor.explore_data()
# Thực hiện khám phá dữ liệu.
    churn_predictor.preprocess_data()
# Thực hiện tiền xử lý dữ liệu.
    churn_predictor.split_data()
# Tách dữ liệu thành tập huấn luyện và tập kiểm tra.
    churn_predictor.train_model()
# Huấn luyện mô hình.
    churn_predictor.evaluate_model()
# Đánh giá mô hình.



if __name__ == "__main__":
    churn_predictor1 = ChurnPrediction('churn-bigml-80.csv')
# Tạo một đối tượng ChurnPrediction với tệp dữ liệu 'churn-bigml-80.csv'.
    churn_predictor2 = ChurnPrediction('churn-bigml-20.csv')
# Tạo một đối tượng ChurnPrediction với tệp dữ liệu 'churn-bigml-20.csv'.

# Xử lý dữ liệu và hiển thị báo cáo cho tệp 1.
    print("Báo cáo cho tệp churn-bigml-80.csv")
    process_data(churn_predictor1)

# Xử lý dữ liệu và hiển thị báo cáo cho tệp 2.
    print("Báo cáo cho tệp churn-bigml-20.csv")
    process_data(churn_predictor2)

from churn_prediction import ChurnPrediction, process_data

def main():

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

if __name__ == "__main__":
    main()


Code up lên đang chỉ là bản ensemble learning vì bản autoML train cần thời gian và khá nặng, kết quả đẩy lên là thu được từ việc train những ngày chẵn(4-6-8-...-60) để tối ưu thời gian nhưng vẫn đảm bảo hiệu suất chấp nhậ được.

Pipe line của Ensemble model:
1. Lấy các feature input và tạo các feature mới đặt trưng, đồng thời fit PolynomialFeature và lưu lại, drop "cứng" các cột "yếu"(đã check)

2. Thực hiện biến đổi Yeo Johnson trên tập train và áp dụng lên tập test, đồng thời lưu lại để sử dụng về sau.

3. Dùng Elasticnet để tạo "dự đoán sơ bộ" cho model chính bằng OOF(out-of-fold) để tránh data leakage.

4. Dùng 3 model là XGBoost, LightGBM và CatBoost để train với dữ liệu đã xử lý(dùng optuna để tối ưu tham số).

5. Build 1 model để tổng hợp dự đoán của 3 model trên nhằm đưa ra kết luận tổng hợp(dùng XGBoost luôn vì nó dễ để custom metric theo NAE).

6. Cuối cùng dựa vào dự đoán của các model, lấy target là chỉ số NAE cuối cùng, train 1 model nữa để có thể hiệu chỉnh được dự đoán của các model trước đó, hạ NAE xuống 0,5-1%

7. Các model theo từng ngày và các phép biến đổi được đóng gói và lưu lại bằng joblib, lúc đánh giá bằng data thật được cung cấp nó sẽ được load lại và sử dụng


Kết quả thu được:

+)Theo quan sát và thử nghiệm thì việc hạn chế các tham số trong optuna của các model để các model có thể tổng quát hóa tốt hơn thì nó đem lại điểm tích cực là NAE trên "dữ liệu thật" giảm đi so với ban đầu, nhưng lại gây ra tiêu cực là NAE "trên tập test" lại tăng lên

+)Trên tập test: NAE từ 4.4%(d4) đến 15.6%(d60) với bộ tham số đã hạn chế (với bộ tham số tối ưu cho tập test thì đạt được 3.5%-14.9%)

+)Trên tập dữ liệu thực(50 mẫu): NAE cho d4 là 2.63% và NAE cho d60 là 7.88%, các ngày ở giữa có biến động tương đối nhưng cũng chỉ dao động từ 4%-8%. Biểu đồ thể hiện được vẽ ở file "Draw.ipynb"

+)Thời gian trung bình cho việc load 1 ngày và đưa ra dự đoán là từ 0.11-0.14 giây cho 1 ngày(chạy đơn luồng)

+)Đối với việc thử nghiệm bằng autoML(autogluon) thì NAE trên tập test đạt được 3.8%(d4)-13.9%(d60) và trên dữ liệu thật là từ 2.1%(d4)-7.8%(d60) với mô hình tốt nhất là "TabPFNv2_r143_BAG_L2" khi chạy "presets="extreme_quality""

Khó khăn gặp phải:

+)Toàn bộ quá trình và tiếp cận đang là xây dựng model cho "từng ngày", chưa tối ưu model tất cả dự báo cho toàn d4-d60

+)autogluon để đạt hiệu suất đủ tốt nhất để báo cáo thì thường mất thời gian khá lâu cho 1 ngày, nên pipeline này sẽ cập nhật sau

+)Chưa có đủ thời gian để chạy sâu pipeline ensemble ở trên nên là có thể chưa đạt được độ ổn định và tối ưu nhất nên các dự đoán có phần hơi nhiễu loạn

Đánh giá:

+)Tuy cùng với bộ dữ liệu được cấp ban đầu được chia train/test, d60 cả trên train và test đều >=12% nhưng khi test trên thực tế thì lại kéo được về 7,88% ==> mô hình đang khái quát tốt mục tiêu

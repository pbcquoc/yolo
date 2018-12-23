# Tìm Hiểu YOLO cho Object Dectection
## Giới Thiệu
You only look once (YOLO) là một mô hình CNN để detect object mà một ưu điểm nổi trội là nhanh hơn nhiều so với những mô hình cũ. Thậm chí có thể chạy tốt trên những IOT device như raspberry pi. Trong phần này mình hướng dẫn các bạn chi tiết cách cài đặt YOLO v1 trên tập dữ liệu mẫu do mình tự phát sinh. Nắm rõ về YOLO v1 giúp bạn có thể cài đặt các phiên bản cải tiến, đồng thời giúp bạn có thể đọc những tài liệu về Object Detection. Đồng thời cung cấp 25k mẫu dữ liệu cho các bạn dùng để thử nghiệm trong bài toán object detection

Mình cung cấp sẵn file ipython giúp các bạn có thể dễ dàng thử nghiệm với mô tả chi tiết từng bước của thuật toán trên tập dữ liệu mẫu.

![object detection example](/image/yolo_example.png)

## Dataset
Mình dùng tập dataset tự phát sinh để làm ví dụ cho thuật toán YOLO v1. Tập dữ liệu này tương đối nhẹ và dễ nên giúp các bạn dễ dàng huấn luyện mô hình với độ chính xác cao đồng thời giúp mình có thể phân tích một số hạn chế của mô hình

![dataset](/image/dataset.png)

## Chi tiết mô hình
Một trong nhưng ưu điểm mà YOLO đem lại đó là chỉ sử dụng thông tin toàn bộ bức ảnh một lần và dự đoán toàn bộ object box chứa các đối tượng, mô hình được xây dựng theo kiểu end-to-end nên được huấn luyện hoàn toàn bằng gradient descent. Sau đây, mình sẽ trình bày chi tiết về mô hình YOLO

### Grid System
Ảnh được chia thành ma trận ô vuông 7x7, mỗi ô vuông bao gồm một tập các thông tin mà mô hình phải dữ đoán.

* Đối tượng duy nhất mà ô vuông đó chứa. Tâm của đối tượng cần xác định nằm trong ô vuông nào thì ô vuông đó chứa đối tượng đó. Ví dụ tâm của cô gái nằm trong ô vuông màu xanh, do đó mô hình phải dự đoán được nhãn của ô vuông đó là cô gái. Lưu ý, cho dù phần ảnh cô gái có nằm ở ô vuông khác mà tâm không thuộc ô vuông đó thì vẫn không tính là chứa cô gái, ngoài ra, nếu có nhiều tâm nằm trong một ô vuông thì chúng ta vẫn chỉ gán một nhãn cho ô vuông đó thôi. Chính ràng buột mỗi ô vuông chỉ chứa một đối tượng là nhược điểm của mô hình này. Nó làm cho ta không thể detect những object có tầm nằm cùng một ô vuông. Tuy nhiên chúng ta có thể tăng grid size từ 7x7 lên kích thước lớn hơn để có thể detect được nhiều object hơn. Ngoài ra, kích thước của ảnh đầu vào phải là bội số của grid size.

![grid_system](https://pbcquoc.github.io/images/yolo_grid_system.png)

* Mỗi ô vuông chịu trách nhiệm dự đoán 2 boundary box của đối tượng. Mỗi boundary box dữ đoán có chứa object hay không và thông tin vị trí của boundary box gồm trung tâm boundary box của đối tượng và chiều dài, rộng của boundary box đó. Ví vụ ô vuông màu xanh cần dự đoán 2 boundary box chứa cô gái như hình minh họa ở dưới. Một điều cần lưu ý, lúc cài đặt chúng ta không dự đoán giá trị pixel mà cần phải chuẩn hóa kích thước ảnh về đoạn từ [0-1] và dự đoán độ lệch của tâm đối tượng đến box chứa đối tượng đó. Ví dụ, chúng ta thay vì dữ đoán vị trí pixel của điểm màu đỏ, thì cần dự đoán độ lệch a,b trong ô vuông chứa tâm object.

![2box](https://pbcquoc.github.io/images/yolo_2box.png)

Tổng hợp lại, với mỗi ô vuông chúng ta cần dữ đoán các thông tin sau :
* Ô vuông có chứa đối tượng nào hay không?
* Dự đoán độ lệch 2 box chứa object so với ô vuông hiện tại
* Lớp của object đó

Như vậy với mỗi ô vuông chúng ta cần dữ đoán một vector có (nbox+4*nbox+nclass) chiều. Ví dụ, chúng ta cần dự đoán 2 box, và 3 lớp đối với mỗi ô vuông thì chúng sẽ có một ma trận 3 chiều 7x7x13 chứa toàn bộ thông tin cần thiết.

![labels](https://pbcquoc.github.io/images/yolo_predict_vector.png)

[See more](https://pbcquoc.github.io/yolo/)

## Kết quả của mô hình
Sau khoảng 100 epochs, các bạn có thể đạt được 0.9 iou trên tập test gồm 5k mẫu dữ liệu. 
![dataset](/image/yolo_train_result.png)

## Any Problems
Nếu bạn có bất kì vấn đề gì, vui lòng liên hệ với mình qua email: pbcquoc@gmail.com

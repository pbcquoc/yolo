# Tìm Hiểu YOLO cho Object Dectection
## Giới Thiệu
You only look once (YOLO) là một mô hình CNN để detect object mà một ưu điểm nổi trội là nhanh hơn nhiều so với những mô hình cũ. Thậm chí có thể chạy tốt trên những IOT device như raspberry pi. Trong phần này mình hướng dẫn các bạn chi tiết cách cài đặt YOLO v1 trên tập dữ liệu mẫu do mình tự phát sinh. Nắm rõ về YOLO v1 giúp bạn có thể cài đặt các phiên bản cải tiến, đồng thời giúp bạn có thể đọc những tài liệu về Object Detection. Đồng thời cung cấp 25k mẫu dữ liệu cho các bạn dùng để thử nghiệm trong bài toán object detection

Mình cung cấp sẵn file ipython giúp các bạn có thể dễ dàng thử nghiệm với mô tả chi tiết từng bước của thuật toán trên tập dữ liệu mẫu.

![object detection example](/image/yolo_example.png)

## Dataset
Mình dùng tập dataset tự phát sinh để làm ví dụ cho thuật toán YOLO v1. Tập dữ liệu này tương đối nhẹ và dễ nên giúp các bạn dễ dàng huấn luyện mô hình với độ chính xác cao đồng thời giúp mình có thể phân tích một số hạn chế của mô hình

![dataset](/image/dataset.png)

## Chi tiết mô hình
Các bạn có thể đọc chi tiết mô hình tại [blog](https://pbcquoc.github.io/yolo/) của mình hoặc xem hướng dẫn trong file ipython. Các nêu khá rõ về hướng tiếp cẫn cũng như chi tiết của mô hình. Đồng thời kết quả trên tập dữ liệu mẫu ở trên

## Kết quả của mô hình
Sau khoảng 100 epochs, các bạn có thể đạt được 0.9 iou trên tập test gồm 5k mẫu dữ liệu. 
![dataset](/image/yolo_train_result.png)

## Any Problems
Nếu bạn có bất kì vấn đề gì, vui lòng liên hệ với mình qua email: pbcquoc@gmail.com
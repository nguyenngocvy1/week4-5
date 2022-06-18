# Nguyễn Ngọc Vỹ
# **WEEK 4,5: Làm quen với mạng Neural nhân tạo ứng dụng cho xử lý ảnh**
# Mục lục:
- [**WEEK 4,5: Làm quen với mạng Neural nhân tạo ứng dụng cho xử lý ảnh**](#week-45-làm-quen-với-mạng-neural-nhân-tạo-ứng-dụng-cho-xử-lý-ảnh)
- [Mục lục:](#mục-lục)
- [Nội dung:](#nội-dung)
  - [**I. Nhiệm vụ:**](#i-nhiệm-vụ)
  - [**II. Phân loại Object Classification, Object localization, Object detection, Instance segmentation:**](#ii-phân-loại-object-classification-object-localization-object-detection-instance-segmentation)
  - [**III. Một số khái niệm quan trọng cho việc ứng dụng mạng neural nhân tạo:**](#iii-một-số-khái-niệm-quan-trọng-cho-việc-ứng-dụng-mạng-neural-nhân-tạo)
  - [**IV. QUEST 1:**](#iv-quest-1)
  - [**V. Object detection on Jetson Inference:**](#v-object-detection-on-jetson-inference)
  - [**VI. QUEST 2:**](#vi-quest-2)
- [Tài liệu tham khảo:](#tài-liệu-tham-khảo)
# Nội dung:
## **I. Nhiệm vụ:**
1. Yêu cầu:
    - Quest 1: Nếu muốn máy bay có khả năng "nhìn" và ra quyết định gần giống con người khi gặp vật cản, các bạn khuyến nghị chúng ta nên sử dụng loại mạng Neural nhân tạo nào?
    - Quest 2: Chạy thử inference object detection trên jetson nano đối với video mẫu (video 1 đính kèm) phát hiện đối tượng người, báo cáo so sánh performance của các model khác nhau mà Jetson Inference hỗ trợ.
    - Quest 3: Retrain model ssd_mobilenet_lite_v2 sao cho model này nhận ra được một đối tượng nữa là cây dừa (ảnh đính kèm), cho xử lý video đính kèm (video 2) và ghi report tương tự quest 2 về performance của mô hình đã train về tốc độ chính x
## **II. Phân loại Object Classification, Object localization, Object detection, Instance segmentation:**
1. Image Classification (phân lớp ảnh):
    - Xác định ảnh đầu vào thuộc nhãn nào.
        1. Input ảnh chỉ chứa 1 đối tượng
        2. Máy tính sẽ đọc ảnh dưới dạng ma trận
        3. Dựa theo việc nhận diện ma trận từ những lần training trước đó để nhận diện nhãn tương ứng với ảnh
        4. Output nhãn tương ứng với ảnh.
        <div align='center'>
        <img src="img\classify.png" width='90%'>
        </div>
    - Bài toán cơ sở cho các bài toán khác trong computer vision.
    - Thách thức bài toán này:
        1. Đa góc nhìn - Viewpoint variation.
        2. Đa dạng về tỉ lệ/ kích thước - Scale variation.
        3. Biến dạng - Deformation.
        4. Bị che khuất - Occlusion
        5. Điều kiện chiếu sáng - Illumination conditions.
        6. Ảnh hưởng bởi bối cảnh - Background clutter.
        7. Đa dạng về biến thể trong một nhãn - Intra-class variation.
2. Object Location (Classification with localization):
    - Sử dụng cơ chế của Image Classification để nhận biết đối tượng.
    - Sau đó xác định vị trí (location) của một đối tượng đang quan tâm trong ảnh thông qua việc bao 1 hình chữ nhật quanh đối tượng(bouding box).
    <div align='center'>
    <img src="img\dog.jpg" width='90%'>
    </div>
3. Object Detection (Phát hiện đối tượng):
    - Xác định vị trí và nhãn dán của tất cả đối tượng trong hình.
    - Input ảnh và danh sách các đối tượng quan tâm => output vị trí các đối tượng kèm theo nhãn dán đối tượng đó.
        <div align='center'>
        <img src="img\car.jpg" width='90%'>
        </div>
    - Ứng dụng: Chú thích hình ảnh (Image Annotation), phát hiện khuôn mặt (Face detection), Nhận diện biển số xe (License Plate Identification), Đếm số người (People counting),...
4. Image Segmentation (Phân đoạn ảnh):
    - Thay vì trả về các bouding box bao quanh các đối tượng quan tâm như Object Detection thì Image Segmentation trả về các đường viền (boudaries) bao quanh đối tượng.
    - Thay vì gán nhãn cho tổng thể đối tượng, Image segmentation gán nhãn chô từng điểm ảnh sao cho các điểm ảnh có cùng nhãn dán thuộc về 1 nhóm.
    - Xác định rõ kích thước và hình dạng của đối tượng.
    - 2 loại bài toán Image Segmentation:
        1. Semantic Segmentation: Phân tách các lớp, các đối tượng cùng 1 lớp thì gom vào 1 nhóm.
        2. Instance Segmentation: Phân tách từng đối tượng trong cùng 1 lớp.
        <div align='center'>
        <img src="img\imgseg.ppm" width='90%'>
        </div>
    - Sử dụng bài toán Semantic Segmentation để phân tách ảnh ra các lớp đối tượng, ở mỗi lớp lại tiếp tục dùng bài toán Instance Segmentation để phân tách các đối tượng trong cùng 1 lớp.
## **III. Một số khái niệm quan trọng cho việc ứng dụng mạng neural nhân tạo:**
1. Train and test dataset:
    - Dataset sẽ bao gồm data [features] và label của data đó, đi theo cặp.
    - Chúng ta sẽ không dùng toàn bộ dataset để train mô hình máy học mà thay vào đó là chúng ta sẽ tách (slipt) cái dataset thành 2 phần:
      1. 80% dataset để train: data_train, label_train
      2. 20% dataset để test: data_valid, label_valid
    - Có thể có các tỉ lệ khác: 70% để train, 30% để test hoặc 90% để train, 10% để test.
    - data_train và label_train, ta sẽ dùng để training cho mô hình máy học
    - data_valid và label_valid là validation, chúng ta sẽ dùng để testing mô hình máy học.
    - Nhập một cái data [features] từ validation vào mô hình máy học. Machine learning model sẽ cung cấp ra biến label predict. Biến lable predict này kết quả của machine learning model tạo nên. Sau đó chúng ta sẽ so sánh cái biến label predict này với label validation để xem 2 cái này nó chênh lệch, sai số với nhau là bao nhiêu. Để điều chỉnh (tinh chỉnh các hyperparameter) và cải thiện mô hình máy học.
    - Tại sao không dùng toàn bộ dataset để train mô hình máy học? Vì nếu dùng toàn bộ như vậy sẽ dẫn đến một trường hợp đó là overfitting. Overfitting có nghĩa là toàn bộ dữ liệu sẽ fit vào mô hình máy học đó và mô hình máy học đó sẽ làm việc rất tốt với những dữ liệu input cũ. Nhưng khi nhập input mới vào, một cái data mới vào machine learning model đó thì lúc này mô hình máy học này sẽ dự đoán rất là tệ. Tại vì nhiều lúc cái input mới đó không cùng cấu trúc với dataset ban đầu. Dẫn đến machine learning chỉ chạy tốt với dataset cũ và chạy rất tệ với data mới.
    <div align='center'>
    <img src="img\test.png" width='90%'>
    </div>  
2. Evaluation:
    - Trong thực tế, 1 bài toán Machine learning có thể được giải quyết bởi nhiều phương pháp, cho ra nhiều mô hình khác nhau. Đứng trước nhiều sự lựa chọn, để tìm ra mô hình phù hợp nhất cho bài toán, chúng ta cần phải **đánh giá hiệu năng của mô hình trên dữ liệu mới**(evaluate model performance on unseen data).
    - Evluation sẽ giúp ta giải đáp những câu hỏi sau:
      - Mô hình đã được huấn luyện thành công hay chưa?
      - Mức độ thành công của mô hình.
      - Khi nào nên dừng quá trình huấn luyện?
      - Khi nào nên cập nhật mô hình?
    - Accuracy (độ chính xác): đánh giá mô hình thường xuyên dự đoán đúng đến mức nào. công thức: accuracy = correct predictions/ total predictions.
    - Tuy nhiên không phải lúc nào độ chính xác cao cũng là tốt. Độ chính xác cao gặp nhiều hạn chế khi sử dụng trên bộ dữ liệu không cân bằng (imbalanced dataset). Hạn chế này xuất phát từ nhược điểm Accuracy chỉ cho ta biết độ chính xác khi dự báo của mô hình mà không thể hiện mô hình đang dự đoán sai như thế nào. Vì vậy chúng ta cần một phương pháp đánh giá khác – Confusion Matrix. 
    - Confusion matrix là một kỹ thuật đánh giá hiệu năng của mô hình cho các bài toán phân lớp. Confusion matrix là một ma trận thể hiện số lượng điểm dữ liệu thuộc vào một class và được dự đoán thuộc vào class.
        <div align='center'>
        <img src="img\evaluation.jpg" width='90%'>
        </div>
    - Ví dụ: xét nghiệm COVID-19 cho 1000 người:
      - Mô hình dự đoán có 30 ca dương tính, trong khi thực tế có 13 người nhiễm COVID-19.
      - Mô hình dự đoán có 970 ca âm tính, nhưng thực tế trong 970 ca đó có 20 ca dương tính.
    - Confusion matrix kết quả dự đoán của mô hình là:
      - True Positive TP = 13: có 13 người nhiễm COVID-19 được mô hình dự đoán đúng.
      - False Positive FP = 17: có 17 người âm tính với COVID-19, nhưng được mô hình dự đoán dương tính.
      - True Negative TN = 950: 950 trường hợp âm tính được mô hình phân loại chính xác.
      - False Negative FN = 20: có 20 trường hợp dương tính với COVID-19 nhưng bị mô hình phân loại sai.
    - Type Error: 
        <div align='center'>
        <img src="img\type error.jpg" width='90%'>
        </div>
    - Với kết quả từ Confusion matrix, chúng ta định lượng độ hiệu quả của mô hình thông qua 2 thang đo phổ biến nhất là Precision và Recall.
    - Precision: trong số các điểm dữ liệu được mô hình phân loại vào lớp Positive, có bao nhiêu điểm dữ liệu thực sự thuộc về lớp Positive.
    -  Recall giúp chúng ta biết được có bao nhiêu điểm dữ liệu thực sự ở lớp Positive được mô hình phân lớp đúng trong mọi điểm dữ liệu thực sự ở lớp Positive.
    - Công thức:
      - precision = TP/(TP + FP)
      - recall = TP/(TP + FN)
    - Precision và Recall có giá trị trong [0,1], hai giá trị này càng gần với 1 thì mô hình càng chính xác. Precision càng cao đồng nghĩa với các điểm được phân loại càng chính xác. Recall càng cao cho thể hiện cho việc ít bỏ sót các điểm dữ liệu đúng.
    -  Tuy nhiên, hai giá trị Precision và Recall thường không cân bằng với nhau (giá trị này tăng thì giá trị kia thường có xu hướng giảm). Để đánh giá cùng lúc cả Precision và Recall, ta sử dụng độ đo F-Score.
    -  Công thức: F(B) = (1 + B^2) * (precision*recall)/(B^2 * precision + recall)
       - β > 1: Recall được coi trọng hơn Precision.
       - β < 1: Precision được coi trọng hơn Recall.
       - β = 1: Precision và Recall được coi trọng ngang nhau.
3. Inference model:
    - Suy luận mô hình học máy là quá trình triển khai mô hình học máy đến môi trường sản xuất để suy ra kết quả từ dữ liệu đầu vào. Tại thời điểm này, mô hình sẽ xử lý dữ liệu đầu vào mới và chưa từng thấy. Khi một mô hình thực hiện suy luận, nó đang tạo ra một kết quả dựa trên thuật toán được đào tạo. Điều này có nghĩa là suy luận mô hình nằm trong giai đoạn triển khai của vòng đời học máy. Các kết quả được suy ra thường được quan sát và theo dõi liên tục, tại thời điểm đó mô hình có thể được đào tạo lại hoặc tối ưu hóa như một giai đoạn riêng biệt của vòng đời mô hình.
    - Suy luận mô hình học máy thường được gọi là chuyển một mô hình vào sản xuất. Do đó, đó là việc sử dụng mô hình cho nhiệm vụ ban đầu được thiết kế để làm. Đó là điểm mà một mô hình học máy sẽ bắt đầu tạo ra lợi tức đầu tư tổng thể của dự án. Đây có thể là bất kỳ số lượng nhiệm vụ và quy trình kinh doanh nào mà các mô hình được sử dụng để tự động hóa hoặc cải thiện, cho dù là phân loại hay nhiệm vụ hồi quy. Những thách thức chính mà quy trình suy luận mô hình phải đối mặt là nhúng mô hình trong kiến ​​trúc hệ thống rộng hơn.
    - Những cân nhắc chính cho suy luận mô hình trong môi trường trực tiếp bao gồm:
      - Lưu lượng dữ liệu của cả dữ liệu đầu vào từ môi trường trực tiếp và dữ liệu đầu ra cho kết quả. Điều này bao gồm điểm nhập cho dữ liệu trực tiếp vào đường ống mô hình.
      - Kiến trúc hệ thống và cách mô hình nhúng trong nó. Điều này có thể là các mô hình học máy chứa chứa từ các tài nguyên hệ thống khác nhau hoặc đường ống dựa trên máy chủ.
      - Chuyển đổi dữ liệu đầu vào từ môi trường trực tiếp thành dữ liệu có thể được xử lý bởi mô hình. Điều này có thể bao gồm một giai đoạn tiền xử lý.
      - Chuyển đổi dữ liệu đầu ra hoặc kết quả thành thông tin có thể hiểu được bởi tổ chức. Ví dụ, một kết quả số từ một mô hình được thiết kế để phát hiện hoạt động gian lận có thể cần phải được chuyển đổi thành các nhãn được xác định có thể được tổ chức hiểu.
## **IV. QUEST 1:**
1. Nếu muốn máy bay có khả năng "nhìn" và ra quyết định gần giống con người khi gặp vật cản, các bạn khuyến nghị chúng ta nên sử dụng loại mạng Neural nhân tạo nào?
2. Có 3 loại Neural network:
   1. Artificial Neural Network (ANN):
      - Loại dữ liệu: dữ liệu dạng bảng, văn bản.
      - Chia sẻ tham số: Không.
      - Cố định chiều dài input: Có.
      - Kết nối thường xuyên: Không.
      - Hiệu năng: ANN kém nhất.
      - Ưu điểm:
        - Lưu trữ thông tin trên toàn bộ mạng.
        - Khả năng làm việc với kiến ​​thức không đầy đủ.
        - Có khả năng chịu lỗi.
        - Có bộ nhớ phân tán.
      - Nhược điểm:
        - Phụ thuộc phần cứng.
        - Hành vi không giải thích được của mạng.
        - Xác định cấu trúc mạng thích hợp.
   2. Convolutional Neural Network (CNN):
      - Loại dữ liệu: dữ liệu hình ảnh.
      - Chia sẻ tham số: Có.
      - Cố định chiều dài input: Có.
      - Kết nối thường xuyên: Không.
      - Hiệu năng: CNN mạnh nhất.
      - Ưu điểm: 
        - Độ chính xác rất cao trong các vấn đề nhận dạng hình ảnh.
        - Tự động phát hiện các tính năng quan trọng mà không có sự giám sát của con người.
        - Chia sẻ khối lượng.
      - Nhược điểm:
        - CNN không mã hóa vị trí và định hướng của vật thể.
        - Thiếu khả năng bất biến không gian đối với dữ liệu đầu vào.
        - Yêu cầu rất nhiều dữ liệu đào tạo.
   3. Recurrent Neural Network (RNN):
      - Dữ liệu có thứ tự, ví dụ: các ảnh được tách từ video, time-series data.
      - Chia sẻ tham số: Có.
      - Cố định chiều dài input: Không.
      - Kết nối thường xuyên: Có.
      - Hiệu năng: kém hơn CNN, tốt hơn ANN.
      - Ưu điểm:
        - Một RNN nhớ từng thông tin theo thời gian. Nó rất hữu ích trong dự đoán chuỗi thời gian chỉ vì tính năng cần ghi nhớ các đầu vào trước đó là tốt. Điều này được gọi là bộ nhớ ngắn hạn dài.
        - Mạng lưới thần kinh tái phát thậm chí còn được sử dụng với các lớp chập để mở rộng vùng lân cận pixel hiệu quả.
      - Nhược điểm: 
        - Gradient vanishing, exploding gradient.
        - Đào tạo RNN là một nhiệm vụ rất khó khăn.
        - Nó không thể xử lý các chuỗi rất dài nếu sử dụng tanh hoặc relu làm chức năng kích hoạt.
3. Kết luận:
   - Theo quan điểm cá nhân của em thì dùng mạng Convolutional Neural Network để tránh vật cản.
   - Vì tận dụng tốt được các ưu điểm như độ chính xác rất cao trong các vấn đề nhận dạng hình ảnh, không cần sự giám sát của con người, hiệu năng cao, không cần phải kết nối thường xuyên, có chia sẻ tham số.
   - Drone bay với độ cao tương đối cao nên những vật drone có thể va trúng cũng chỉ bao gồm một số vật như cột điện, dây điện, tòa nhà,.. việc ít class nên sẽ giảm thiểu được rất nhiều điểm yếu cần nhiều dữ liệu để đào tạo của mạng CNN.
   - Lý do không chọn mạng RNN: Vì đào tạo RNN là một nhiệm vụ rất khó khăn.
   - Lý do không chọn mạng ANN: đây là mạng tệ nhất trong 3 mạng, phụ thuộc vào phần cứng. Một chức 
## **V. Object detection on Jetson Inference:**
- Download 2 thư viện:
  1. jetson-inference.
  2. jetson-utils.
- Module detectnet.py có sẵn trong thư viện jetson-inference
- Chọn network, input ảnh hoặc video vào, output thu được là ảnh hoặc video đã được detect object(phát hiện nhiều đối tượng, bouding box từng đối tượng, hiện nhãn dán đối tượng, và giá trị tin cậy của đối tượng đó đối với nhãn dán)
- Một số Pre-trained Detection Models có sẵn:
    |Model	|CLI argument	|NetworkType enum	|Object classes|
    |:------|:------------|:----------------|:-------------|
    |SSD-Mobilenet-v1	|ssd-mobilenet-v1	|SSD_MOBILENET_V1	|91 (COCO classes)|
    |SSD-Mobilenet-v2	|ssd-mobilenet-v2	|SSD_MOBILENET_V2	|91 (COCO classes)|
    |SSD-Inception-v2	|ssd-inception-v2	|SSD_INCEPTION_V2	|91 (COCO classes)|
    |DetectNet-COCO-Dog	|coco-dog	|COCO_DOG	|dogs
    |DetectNet-COCO-Bottle	|coco-bottle	|COCO_BOTTLE	|bottles|
    |DetectNet-COCO-Chair	|coco-chair	|COCO_CHAIR	|chairs|
    |DetectNet-COCO-Airplane	|coco-airplane	|COCO_AIRPLANE	|airplanes|
    |ped-100	|pednet	|PEDNET	|pedestrians|
    |multiped-500	|multiped	|PEDNET_MULTI	|pedestrians, luggage|
    |facenet-120	|facenet	|FACENET	|faces|
- Cú pháp: ./detectnet.py --network={CLI argument} {input path and file name} {output path and file name}
- Ví dụ: ./detectnet.py --network=ssd-inception-v2 input.jpg output.jpg
## **VI. QUEST 2:**
1. Chạy thử inference object detection trên jetson nano đối với video mẫu (video 1 đính kèm) phát hiện đối tượng người, báo cáo so sánh performance của các model khác nhau mà Jetson Inference hỗ trợ.
    <div align='center'>
    <img src="frame\frame18.jpg" width='90%'>
    </div>
2. Các bước tiếp cận:
   - Mục tiêu: nhận diện người trong video.
   - Đối tượng: người (person)
   - Loại nhận diện: Object Detection.
   - Thư viện jetson-inference.
   - Module: detectnet.py
   - Các network sử dụng: 
        |Model	|CLI argument	|NetworkType enum	|Object classes|
       |:------|:------------|:----------------|:-------------|
       |SSD-Mobilenet-v1	|ssd-mobilenet-v1	|SSD_MOBILENET_V1	|91 (COCO classes)|
       |SSD-Mobilenet-v2	|ssd-mobilenet-v2	|SSD_MOBILENET_V2	|91 (COCO classes)|
       |SSD-Inception-v2	|ssd-inception-v2	|SSD_INCEPTION_V2	|91 (COCO classes)|
       |ped-100	|pednet	|PEDNET	|pedestrians|
       |multiped-500	|multiped	|PEDNET_MULTI	|pedestrians, luggage|
    - Sử dụng Model Downloader tool (download-models.sh) để download 5 model network có sẵn trên:
      - SSD-Moblilenet-v1:
        <div align='center'>
        <img src="img\mobilev1.jpg" width='90%'>
        </div>
      - SSD-Mobilenet-v2:
        <div align='center'>
        <img src="img\mobilev2.jpg" width='90%'>
        </div> 
      - incepion:
        <div align='center'>
        <img src="img\inception.jpg" width='90%'>
        </div>
      - pednet-100:
        <div align='center'>
        <img src="img\pednet.jpg" width='90%'>
        </div>
      - multiped-500:
        <div align='center'>
        <img src="img\multiped.jpg" width='90%'>
        </div>
    - vào cmd nhập "jtop" để xem thông số load của CPU, RAM, GPU của jetson.
    - Code tách video thành frame:
    ```python
    import cv2

    video_path = 'RT Robotics\\intern\\week4-5.mp4'

    vidcap = cv2.VideoCapture(video_path)
    success,frame = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("week4-5\\frame\\frame%d.jpg" % count, frame)     # save frame as JPEG file      
        success,frame = vidcap.read()
        print('Read a new frame%d: ' % count, success)
        count += 1
    ```
    - Từ các video đã xử lí em tách thành các frame ảnh để đánh giá kết quả xử lý của các mô hình mạng (sử dụng ít nhất 10 ảnh cho 1 tập dữ liệu) 
    - Bảng so sánh:
      |Evaluation|SSD-Mobilenet-v1|SSD-Mobilenet-v2|SSD-Inception-v2|ped-100|multiped-500|
      |:---------|:---------------|:---------------|:---------------|:------|:-----------|
      |Accuracy  |99.99%          |99.99%          |99.99%          |99.99% |99.99%      |
      |Precision |22%             |19.61%          |25.49%          |30.38% |27.49%      |
      |Recall    |85.71%          |90%             |92.86%          |100%   |96.86%      |
      |FPS       |30              |23              |18              |8      |9           |
    - Đánh giá và nhận xét:
      - Những model cho tốc độ xử lý ảnh nhanh, nhiều fps thì khả năng nhận diện vật và độ chính xác thấp.
      - Ngược lại những model cho khả năng nhận diện vô cùng chính xác thì fps lại cực kì thấp.
      - Ngoài do tốc độ xử lý ảnh của module, việc fps thấp còn do 2 yếu tố:
        1. Do cấu hình của Jetson Nano: khi chạy máy đã gần như full load GPU (load 99% GPU với tốc độ 921MHz) khi chạy. Nếu GPU tốt hơn thì máy đã có thể cho ra và hiển thị được nhiều fps hơn.
        2. Do fps của video input (điều này chỉ xảy ra khi dùng cấu hình rất mạnh và video input có fps quá thấp). Ví dụ: xài GPU cực khủng + module SSD-Mobilenet mà detect video đầu vào 15fps thì cùng lắm cũng chỉ thu lại được video 15fps.
        3. Bonus: Do khả năng hiển thị của màn hình (điều này rất hiếm xảy ra vì hầu hết màn hình hiện tại đã có giá thành rất rẻ trong khi có tốc độ làm mới rất cao từ 60Hz, 75Hz, 90Hz, 144Hz, 165Hz,... trong khi module chúng ta chỉ cần xử lý ảnh cho ra 60fps là đã rất thành công.) Tốc độ refresh của màn hình chỉ ảnh hưởng fps khi ta xem, không ảnh hưởng fps của video trong file.
      - ped-100 chuyên về nhận diện người đi bộ nên nhận diện người vô cùng tốt, tốt nhất trong các module với khả năng recall 100%. Do module này chỉ để nhận diện người nên không bị lầm người với các vật thể khác, trong khi các module khác lầm trẻ em thành hành lý, lầm chòi thành chiếc dù, lầm tảng đá thành ô tô.
      - multiped-500 chuyên về nhận diện người và hành lý cho khả năng nhận diện người cũng rất tốt nhưng thỉnh thoảng nhầm giữa trẻ em đi cùng người lớn thành hành lý.
# Tài liệu tham khảo:
1. http://tutorials.aiclub.cs.uit.edu.vn/index.php/2020/04/28/phan-biet-bai-toan-trong-cv/?fbclid=IwAR09N4yNjlOEp8XIlhgwcbEsMlf5Fe7wKkxSq93Rq3JeabqsdVnJxgcxAII
2. https://cs231n.github.io/classification/
3. https://machinelearningmastery.com/applications-of-deep-learning-for-computer-vision/
4. https://www.jeremyjordan.me/evaluating-a-machine-learning-model/
5. https://github.com/dusty-nv/jetson-inferencehttps://github.com/dusty-nv/jetson-inference
6. https://www.youtube.com/watch?v=bcM5AQSAzUY
7. https://www.forecr.io/blogs/ai-algorithms/hello-ai-world
8. https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-example-2.md
9. https://www.codegrepper.com/code-examples/python/python+split+video+into+frames
10. http://tutorials.aiclub.cs.uit.edu.vn/index.php/2021/05/18/evaluation/
11. https://www.seldon.io/machine-learning-model-inference-vs-machine-learning-training#:~:text=In%20contrast%2C%20model%20inference%20is,model%20to%20produce%20a%20result.
12. https://www.geeksforgeeks.org/difference-between-ann-cnn-and-rnn/#:~:text=ANN%20is%20considered%20to%20be,compatibility%20when%20compared%20to%20CNN.&text=Facial%20recognition%20and%20Computer%20vision.

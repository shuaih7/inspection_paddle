## 测试预训练模型

### 测试单张图片文字检测结果
    python PaddleOCR-release-2.0-rc1-0/tools/infer_det.py -c ./det_mv3_db.yml -o Global.infer_img="./sample.jpg" Global.pretrained_model="E:\Projects\Part_Number\model\ch_ppocr_mobile_v2.0_det_train\best_accuracy" Global.load_static_weights=false
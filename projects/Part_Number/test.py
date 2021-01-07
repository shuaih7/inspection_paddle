from paddleocr import PaddleOCR, draw_ocr
# Paddleocr??????????????????????,??????lang??????
# ?????`ch`, `en`, `french`, `german`, `korean`, `japan`?
ocr = PaddleOCR(use_angle_cls=True, lang="ch") # need to run only once to download and load model into memory
img_path = './sample.jpg'
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)
 
# ????
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save("./results/sample_result.jpg")

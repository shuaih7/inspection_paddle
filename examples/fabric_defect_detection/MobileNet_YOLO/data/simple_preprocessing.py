import cv2
import glob as gb
import os

def side_chop(img_load_path, img_save_path):
    img_list = gb.glob(img_load_path + r"/*.bmp")
    for i, img_file in enumerate(img_list):
        img = cv2.imread(os.path.join(img_load_path, img_file))
        h, w = img.shape[:2]
        chop_width = int((w - h) / 2)
        img = img[:, chop_width:(w - chop_width)]
        img = cv2.resize(img, (352, 352))
        output_name = 'valid_' + str(i) + '.png'
        cv2.imwrite(os.path.join(img_save_path, output_name), img)
        print('image %d chopped and resized.' % i)

if __name__ == '__main__':
    img_load_path = r'E:\Projects\Fabric_Defect_Detection\ThreeGun_1013\sampling_1013_40hz_white'
    img_save_path = r'E:\Projects\Fabric_Defect_Detection\model_proto\ShuffleNetV2_YOLOv3\v1.0.1\dataset\valid_white'
    side_chop(img_load_path, img_save_path)
import mindspore.dataset as ds
import cv2
from PIL import Image
file_name ='/home/data/wangll22/mmocr-main/mmocr-main/data/wildreceipt/image_files/Image_1/0/01e19e8b2d86190ca474aaaa8f19d44ebaf469be.jpeg'

#data = Image.open(file_name)
data = cv2.imread(file_name)
print(data)
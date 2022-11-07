import numpy as np
import mindspore.dataset as ds
import cv2
from fileio import _from_file_getfilename, _from_file_getheight, _from_file_getwidth, _from_file_getannation,_from_file_train_id
import os
data_root = '/home/data/wangll22/mmocr-main/mmocr-main/'
image_root = '/home/data/wangll22/mmocr-main/mmocr-main/data/wildreceipt/'

ann_file = '/home/data/wangll22/mmocr-main/mmocr-main/data/wildreceipt/train.txt'
dict_file = 'data/wildreceipt/dict.txt'
def sdmgr_dataset():
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = eval(line)
            image = cv2.imread(image_root + line['file_name'])
            image = image.shape
            yield np.array([image]),line["annotations"]
       # filename = np.asarray(filename)
      #  annation = np.asarray(annation)

dataset = ds.GeneratorDataset(list(sdmgr_dataset()), column_names=['filename','annation'])
for data in dataset.create_dict_iterator():
    print("filename:[{:9.5f}]  ".format(data('filename')))
print("data size:", dataset.get_dataset_size())




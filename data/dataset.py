import numpy as np
import mindspore.dataset as ds

from fileio import _from_file_getfilename, _from_file_getheight, _from_file_getwidth, _from_file_getannation,_from_file_train_id
import os
data_root = '/home/data/wangll22/mmocr-main/mmocr-main/'
image_root = '/home/data/wangll22/mmocr-main/mmocr-main/data/wildreceipt/image_files'

ann_file = '/home/data/wangll22/mmocr-main/mmocr-main/data/wildreceipt/train.txt'
dict_file = 'data/wildreceipt/dict.txt'
class SDMGRdataset:
    def __int__(self):
       self.data_dir = data_root
       self.image_dir = image_root
       self.train_id = _from_file_train_id(ann_file)
    def __getitem__(self, index):
        return self._from_file_getfilename(ann_file)[index], self._from_file_getheight(ann_file)[index], \
               self._from_file_getwidth(ann_file)[index], self._from_file_getannation(ann_file)[index]

    def __len__(self):
        """自定义获取样本数据量函数"""
        return len(self.train_id)


dataset_generator = SDMGRdataset()
dataset = ds.GeneratorDataset(dataset_generator, ["data","annation"], shuffle=False)

for data in dataset.create_dict_iterator():
    data1 = data['data'].asnumpy()
    height1 = data['height'].asnumpy()
    print(data1)
print("1")


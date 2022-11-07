import os
import mindspore.dataset as ds
import mindspore.dataset.text as text

data_file ='/home/data/wangll22/mmocr-main/mmocr-main/data/wildreceipt/train.txt'
dataset = ds.TextFileDataset(data_file, shuffle=False)
for data in dataset.create_dict_iterator(output_numpy=True):
    print(text.to_str(data['text']))
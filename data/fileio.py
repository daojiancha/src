
import numpy
from PIL import Image
def _from_file_getfilename(ann_file):

    with open(ann_file, 'r', encoding='utf-8') as f:
        filename = []
        for line in f.readlines():
            line = eval(line)
            data_root =  '/home/data/wangll22/mmocr-main/mmocr-main/data/wildreceipt/'
            data = numpy.array(Image.open(data_root+line["file_name"]))
            filename.append(data)

    return filename

def _from_file_getheight(ann_file):

    with open(ann_file, 'r', encoding='utf-8') as f:
        height =[]
        for line in f.readlines():
            line = eval(line)
            height.append(line["height"])
    return height

def _from_file_getwidth(ann_file):

    with open(ann_file, 'r', encoding='utf-8') as f:
        width =[]
        for line in f.readlines():
            line = eval(line)
            width.append(line["width"])
    return width


def _from_file_getannation(ann_file):

    with open(ann_file, 'r', encoding='utf-8') as f:
        annation = []
        for line in f.readlines():
            line = eval(line)
            annation.append(line["annotations"])
    return annation

def list_from_file(filename, encoding='utf-8'):
    """Load a text file and parse the content as a list of strings. The
    trailing "\\r" and "\\n" of each line will be removed.

    Note:
        This will be replaced by mmcv's version after it supports encoding.

    Args:
        filename (str): Filename.
        encoding (str): Encoding used to open the file. Default utf-8.

    Returns:
        list[str]: A list of strings.
    """
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for line in f:
            item_list.append(line.rstrip('\n\r'))
    return item_list

def data_info(ann_file):
    with open(ann_file, 'r', encoding='utf-8') as f:
        data =[]
        for line in f.readlines():
            line = eval(line)
            data.append(line)

    return data

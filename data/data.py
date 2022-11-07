from statistics import mode
import numpy as np
from os import path as osp
from box_utils import sort_vertex8
import warnings
from fileio import _from_file_getfilename, _from_file_getheight, _from_file_getwidth, _from_file_getannation,list_from_file,data_info
from mindspore import dataset as ds
import collections

ann_file='/home/data/wangll22/mmocr-main/mmocr-main/data/wildreceipt/train.txt'
class SDMGRdataset():

    def __init__(self,
                 ann_file='/home/data/wangll22/mmocr-main/mmocr-main/data/wildreceipt/train.txt',
                 loader=None,
                 dict_file='/home/data/wangll22/mmocr-main/mmocr-main/data/wildreceipt/dict.txt',
                 img_prefix='',
                 norm=10.,
                 pipeline = None,
                 directed=False,
                 **kwargs):
        if ann_file is None and loader is None:
            warnings.warn(
                'KIEDataset is only initialized as a downstream demo task '
                'of text detection and recognition '
                'without an annotation file.', UserWarning)
        else:
            super().__init__(
            )
            assert osp.exists(dict_file)
        self.img_prefix = img_prefix
        self.norm = norm
        self.directed = directed
        self.data = data_info(ann_file=ann_file)
        self.dict = {
            '': 0,
            **{
                line.rstrip('\r\n'): ind
                for ind, line in enumerate(list_from_file(dict_file), 1)
            }
        }

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []
        results['ori_texts'] = results['ann_info']['ori_texts']
        results['filename'] = osp.join(self.img_prefix,
                                       results['img_info']['filename'])
        results['ori_filename'] = results['img_info']['filename']
        # a dummy img data
        results['img'] = np.zeros((0, 0, 0), dtype=np.uint8)

    def _parse_anno_info(self, annotations):
        """Parse annotations of boxes, texts and labels for one image.
        Args:
            annotations (list[dict]): Annotations of one image, where
                each dict is for one character.

        Returns:
            dict: A dict containing the following keys:

                - bboxes (np.ndarray): Bbox in one image with shape:
                    box_num * 4. They are sorted clockwise when loading.
                - relations (np.ndarray): Relations between bbox with shape:
                    box_num * box_num * D.
                - texts (np.ndarray): Text index with shape:
                    box_num * text_max_len.
                - labels (np.ndarray): Box Labels with shape:
                    box_num * (box_num + 1).
        """

        assert len(annotations) > 0, 'Please remove data with empty annotation'
        assert 'box' in annotations[0]
        assert 'text' in annotations[0]

        boxes, texts, text_inds, labels, edges = [], [], [], [], []
        for ann in annotations:
            box = ann['box']
            sorted_box = sort_vertex8(box[:8])
            boxes.append(sorted_box)
            text = ann['text']
            texts.append(ann['text'])
            text_ind = [self.dict[c] for c in text if c in self.dict]
            text_inds.append(text_ind)
            labels.append(ann.get('label', 0))
            edges.append(ann.get('edge', 0))

        ann_infos = dict(
            boxes=boxes,
            texts=texts,
            text_inds=text_inds,
            edges=edges,
            labels=labels)

        return self.list_to_numpy(ann_infos)
    def prepare_train_img(self,ann_file, index):
        self.data = data_info(ann_file=ann_file)
        img_ann_info = self.data[index]
        img_info = {
            'filename': img_ann_info['file_name'],
            'height': img_ann_info['height'],
            'width': img_ann_info['width']
        }
        ann_info = self._parse_anno_info(img_ann_info['annotations'])

        results = dict(img_info=img_info, ann_info=ann_info)
        print(results)
        return results

    def list_to_numpy(self, ann_infos):
        """Convert bboxes, relations, texts and labels to ndarray."""
        boxes, text_inds = ann_infos['boxes'], ann_infos['text_inds']
        texts = ann_infos['texts']
        boxes = np.array(boxes, np.int32)
        relations, bboxes = self.compute_relation(boxes)

        labels = ann_infos.get('labels', None)
        if labels is not None:
            labels = np.array(labels, np.int32)
            edges = ann_infos.get('edges', None)
            if edges is not None:
                labels = labels[:, None]
                edges = np.array(edges)
                edges = (edges[:, None] == edges[None, :]).astype(np.int32)
                if self.directed:
                    edges = (edges & labels == 1).astype(np.int32)
                np.fill_diagonal(edges, -1)
                labels = np.concatenate([labels, edges], -1)
        padded_text_inds = self.pad_text_indices(text_inds)

        return dict(
            bboxes=bboxes,
            relations=relations,
            texts=padded_text_inds,
            ori_texts=texts,
            labels=labels)

    def pad_text_indices(self, text_inds):
        """Pad text index to same length."""
        max_len = max([len(text_ind) for text_ind in text_inds])
        padded_text_inds = -np.ones((len(text_inds), max_len), np.int32)
        for idx, text_ind in enumerate(text_inds):
            padded_text_inds[idx, :len(text_ind)] = np.array(text_ind)
        return padded_text_inds

    def compute_relation(self, boxes):
        """Compute relation between every two boxes."""
        # Get minimal axis-aligned bounding boxes for each of the boxes
        # yapf: disable
        bboxes = np.concatenate(
            [boxes[:, 0::2].min(axis=1, keepdims=True),
             boxes[:, 1::2].min(axis=1, keepdims=True),
             boxes[:, 0::2].max(axis=1, keepdims=True),
             boxes[:, 1::2].max(axis=1, keepdims=True)],
            axis=1).astype(np.float32)
        # yapf: enable
        x1, y1 = bboxes[:, 0:1], bboxes[:, 1:2]
        x2, y2 = bboxes[:, 2:3], bboxes[:, 3:4]
        w, h = np.maximum(x2 - x1 + 1, 1), np.maximum(y2 - y1 + 1, 1)
        dx = (x1.T - x1) / self.norm
        dy = (y1.T - y1) / self.norm
        xhh, xwh = h.T / h, w.T / h
        whs = w / h + np.zeros_like(xhh)
        relation = np.stack([dx, dy, whs, xhh, xwh], -1).astype(np.float32)
        return relation, bboxes

    def __getitem__(self,index):
        """Get training/test data from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training/test data.
        """
        data = self.prepare_train_img(ann_file,index)
        return data

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.__index >= len(self.data):
            raise StopIteration
        else:
            item = self.prepare_train_img(ann_file,self.__index)
            self.__index += 1
            return item

    def __iter__(self):
        self.__index = 0
        return self
dataset_generator = SDMGRdataset()
'''data = []
for i in range(dataset_generator.__len__()):
    data.append(SDMGRdataset().prepare_train_img(ann_file,i))

dataset = ds.NumpySlicesDataset(data,["img_info","ann_info"], shuffle=False)'''

#for data in dataset.create_dict_iterator():
    #print(data['img_info'])
#print("data size:", dataset.get_dataset_size())
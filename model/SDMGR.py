import mindspore.nn as nn
import mindspore.ops as ops
from Block import Block
from Layer import GNNLayer
from backbone import UNet
from bbox2roi import bbox2roi
class SDMGR(nn.Cell):
    def __init__(self,
                 num_chars=92,
                 visual_dim=64,
                 fusion_dim=1024,
                 node_input=32,
                 node_embed=256,
                 edge_input=5,
                 edge_embed=256,
                 num_gnn=2,
                 num_classes=26,
                 visual_modality = False,
                 backbone = None,
                 bidirectional=False,
                 ):
        '''self.maxpool = nn.MaxPool2d(kernel_size=7, stride=7)
        self.backbone = UNet(visual_dim, num_classes, fusion_dim, use_deconv=True, use_bn=True)
        if visual_modality:
            self.extractor =self.backbone
            self.maxpool = self.maxpool(7)
            self.roi = ops.ROIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=0, aligned=True, use_torchvision=False)
        else:
            self.extractor = None'''
        # 文本与视觉信息融合模块'
        super(SDMGR,self).__init__()
        self.node_embed = nn.Embedding(num_chars, node_input, use_one_hot=False)
        hidden = node_embed // 2 if bidirectional else node_embed

        # 单层lstm
        self.rnn = nn.LSTM(
            input_size=node_input,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional)
        # 图推理模块
        self.edge_embed = nn.Dense(edge_input, edge_embed)

        self.gnn_layers = nn.CellList(
        [GNNLayer(node_embed, edge_embed) for _ in range(num_gnn)])
        # 分类模块
        self.node_cls = nn.Dense(node_embed, num_classes)
        self.edge_cls = nn.Dense(edge_embed, 2)
        #self.loss = nn.CrossEntropyLoss(loss)

#def construct(self,inputs):

    def extract_feat(self, img, gt_bboxes):
        if self.visual_modality:
         # 视觉特征提取
            x = super().extract_feat(img)[-1]
            feats = self.maxpool(self.extractor([x], bbox2roi(gt_bboxes)))
            return feats.view(feats.size(0), -1)
        return None

    def construct(self, relations, texts,x=None):
     # relation是节点之间关系编码，shape为[batch,文本框个数，文本框个数，5]，其中这个5是固定的，代表上文的公式7-9对应的值
     # texts是文本信息,shape为[batch,文本框个数,文本框中字符最大值]
     # x是图特征 img,gt_bboxes,gt_labels
        #x = self.extract_feat(img, gt_bboxes)
        self.fusion = Block([visual_dim, node_embed], node_embed, fusion_dim)
        node_nums, char_nums = [], []
        for text in texts:
            node_nums.append(text.size(0))
            char_nums.append((text > 0).sum(-1))
    # 取出一批数据中的最长文本的长度
        max_num = max([char_num.max() for char_num in char_nums])

# 进行padding操作
        concat_op = ops.Concat()
        cast_op = ops.Cast()
        all_nodes = concat_op(cast_op([
        concat_op(cast_op(
            [text,text.new_zeros(text.size(0), max_num - text.size(1))], -1))for text in texts
            ]))

    # 编码文本信息
        embed_nodes = self.node_embed(all_nodes.clamp(min=0).long())
        rnn_nodes, _ = self.rnn(embed_nodes)

        nodes = rnn_nodes.new_zeros(*rnn_nodes.shape[::2])
        all_nums = concat_op(cast_op(char_nums))
        valid = all_nums > 0
        nodes[valid] = rnn_nodes[valid].gather(1, (all_nums[valid] - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, rnn_nodes.size(-1))).squeeze(1)

    # 视觉特征和文本特征融合
        if x is not None:
            nodes = self.fusion([x, nodes])

# 图推理模块
# 根据输入的两个文本框之间的空间位置关系，对边关系进行编码（重要影响）
        all_edges = concat_op([rel.view(-1, rel.size(-1)) for rel in relations])
        embed_edges = self.edge_embed(all_edges.float())
        embed_edges = ops.L2Normalize(embed_edges)

        for gnn_layer in self.gnn_layers:
        # 这里输入虽然是batch，但是输出的时候把batch的结果拼接到一起了
        # nodes.shape = [sum(batch_box_num),256]
        # cat_nodes.shape = [sum(batch_box_num^2),256]
            nodes, cat_nodes = gnn_layer(nodes, embed_edges, node_nums)

    # 多分类模块
    # node_cls.shape = [sum(batch_box_num),label_num]
    # edge_cls .shape = [sum(batch_box_num^2),2]
        node_cls, edge_cls = self.node_cls(nodes), self.edge_cls(cat_nodes)
        return node_cls, edge_cls

'''def extract_feat(self, img, gt_bboxes):
        if self.visual_modality:
		# 视觉特征提取
            x = super().extract_feat(img)[-1]
            feats = self.maxpool(self.extractor([x], bbox2roi(gt_bboxes)))
            return feats.view(feats.size(0), -1)
        return None'''


    
model = SDMGR()
print(model.trainable_params())

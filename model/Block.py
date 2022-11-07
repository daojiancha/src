import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms 
import numpy as np
class Block(nn.Cell):

    def __init__(self,
                 input_dims,
                 output_dim,
                 mm_dim=1600,
                 chunks=20,
                 rank=15,
                 shared=False,
                 dropout_input=0.,
                 dropout_pre_lin=0.,
                 dropout_output=0.,
                 pos_norm='before_cat'):
        super().__init__()
        self.rank = rank
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert (pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Dense(input_dims[0], mm_dim)
        self.linear1 = (
            self.linear0 if shared else nn.Dense(input_dims[1], mm_dim))
        self.merge_linears0 = nn.CellList()
        self.merge_linears1 = nn.CellList()
        self.chunks = self.chunk_sizes(mm_dim, chunks)
        for size in self.chunks:
            ml0 = nn.Dense(size, size * rank)
            self.merge_linears0.append(ml0)
            ml1 = ml0 if shared else nn.Dense(size, size * rank)
            self.merge_linears1.append(ml1)
        self.linear_out = nn.Dense(mm_dim, output_dim)
        self.spilt = ops.Split()
        self.reducesum = ops.ReduceSum()
        self.sqrt = ops.Sqrt()
        self.relu = ops.ReLU()
        self.normalize = ops.L2Normalize()
        self.cat = ops.Concat()
        self.dropout = ops.Dropout()
    def construct(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bs = x1.size(0)
        if self.dropout_input > 0:
            x0 = nn.Dropout(keep_prob=1-self.dropout_input)
            x0.set_train()
            x1 = nn.Dropout(keep_prob=1-self.dropout_input)
            x1.set_train()
        x0_chunks = self.spilt(x0, self.chunks, -1)
        x1_chunks = self.spilt(x1, self.chunks, -1)
        zs = []
        for x0_c, x1_c, m0, m1 in zip(x0_chunks, x1_chunks,
                                      self.merge_linears0,
                                      self.merge_linears1):
            m = m0(x0_c) * m1(x1_c)  # bs x split_size*rank
            m = m.view(bs, self.rank, -1)
            z = self.reducesum(m, 1)
            if self.pos_norm == 'before_cat':
                z = self.sqrt(self.relu(z)) - self.sqrt(self.relu(-z))
                z = self.normalize(z)
            zs.append(z)
        z = self.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = self.sqrt(self.relu(z)) - self.sqrt(self.relu(-z))
            z = self.normalize(z)

        if self.dropout_pre_lin > 0:
            z = self.dropout(keep_prob=1-self.dropout_pre_lin)
            z.set_train()
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z =self.dropout( keep_prob=1-self.dropout_output)
            z.set_train()
        return z
    @staticmethod
    def chunk_sizes(dim, chunks):
        split_size = (dim + chunks - 1) // chunks
        sizes_list = [split_size] * chunks
        sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
        return sizes_list
'''ms.set_context(mode=ms.PYNATIVE_MODE)
test_net = Block([64,256],256,1024)
x = ms.Tensor(np.random.randint(0, 10, [2, 256, 1024, 512]), ms.float32)
x= [x,x]
out = test_net(x)
print(out)
'''

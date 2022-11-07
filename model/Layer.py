import mindspore.nn as nn
import mindspore.ops as ops

class GNNLayer(nn.Cell):

    def __init__(self, node_dim=256, edge_dim=256):
        super().__init__()
        self.in_fc = nn.Dense(node_dim * 2 + edge_dim, node_dim)
        self.coef_fc = nn.Dense(node_dim, 1)
        self.out_fc = nn.Dense(node_dim, node_dim)
        self.relu = nn.ReLU()

    def forward(self, nodes, edges, nums):
        start, cat_nodes = 0, []
        for num in nums:
            sample_nodes = nodes[start:start + num]
            cat_nodes.append(
                ops.Concat([
                    sample_nodes.unsqueeze(1).expand(-1, num, -1),
                    sample_nodes.unsqueeze(0).expand(num, -1, -1)
                ], -1).view(num ** 2, -1))
            start += num
        # 公式11
        cat_nodes = ops.Concat([ops.Concat(cat_nodes), edges], -1)
        # 公式12-13
        cat_nodes = self.relu(self.in_fc(cat_nodes))
        coefs = self.coef_fc(cat_nodes)

        # 公式14
        start, residuals = 0, []
        for num in nums:
            residual = ops.Softmax(
                -ops.Eye(num).to(coefs.device).unsqueeze(-1) * 1e9 +
                coefs[start:start + num ** 2].view(num, num, -1), 1)
            residuals.append(
                (residual *
                 cat_nodes[start:start + num ** 2].view(num, num, -1)).sum(1))
            start += num ** 2

        nodes += self.relu(self.out_fc(ops.Concat(residuals)))
        return nodes, cat_nodes

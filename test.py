from re import I
import mindspore as ms
import numpy as np
x = np.random.uniform(-1, 1, (2, 1, 12, 12)).astype(np.float32)
x1 = ms.Tensor(1.2,dtype=ms.float16)
print(x)
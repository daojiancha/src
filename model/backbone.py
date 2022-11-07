# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Model of UnetPlusPlus


from mindspore import Tensor, float16, float32, int64, int8
import mindspore.ops as P
from numpy import inner, int32
from torch import tensor
""" Parts of the U-Net-PlusPlus model """
import mindspore.nn as nn
import mindspore.ops.functional as F
import mindspore.ops.operations as P
import numpy as np
import mindspore as ms
from mindspore import context
def conv_bn_relu(in_channel, out_channel, use_bn=True, kernel_size=3, stride=1, pad_mode="same", activation='relu'):
    output = []
    output.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode=pad_mode))
    if use_bn:
        output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)


class UnetConv2d(nn.Cell):
    """
    Convolution block in Unet, usually double conv.
    """
    def __init__(self, in_channel, out_channel, use_bn=True, num_layer=2, kernel_size=3, stride=1, padding='same'):
        super(UnetConv2d, self).__init__()
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channel = in_channel
        self.out_channel = out_channel

        convs = []
        for _ in range(num_layer):
            convs.append(conv_bn_relu(in_channel, out_channel, use_bn, kernel_size, stride, padding, "relu"))
            in_channel = out_channel

        self.convs = nn.SequentialCell(convs)

    def construct(self, inputs):
        x = self.convs(inputs)
        return x


class UnetUp(nn.Cell):
    """
    Upsampling high_feature with factor=2 and concat with low feature
    """
    def __init__(self, in_channel, out_channel, use_deconv, n_concat=2):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2d(in_channel + (n_concat - 2) * out_channel, out_channel, False)
        self.concat = P.Concat(1)
        self.use_deconv = use_deconv
        if use_deconv:
            self.up_conv = nn.Conv2dTranspose(in_channel, out_channel, kernel_size=2, stride=2, pad_mode="same")
        else:
            self.up_conv = nn.Conv2d(in_channel, out_channel, 1)

    def construct(self, high_feature, *low_feature):
        if self.use_deconv:
            output = self.up_conv(high_feature)
        else:
            _, _, h, w = F.shape(high_feature)
            output = P.ResizeBilinear((h * 2, w * 2))(high_feature)
            output = self.up_conv(output)
        for feature in low_feature:
            output = self.concat((output, feature))
        return self.conv(output)

class NestedUNet(nn.Cell):
    """
    Nested unet
    """
    def __init__(self, in_channel, n_class=2, feature_scale=2, use_deconv=True, use_bn=True, use_ds=True):
        super(NestedUNet, self).__init__()
        self.in_channel = in_channel
        self.n_class = n_class
        self.feature_scale = feature_scale
        self.use_deconv = use_deconv
        self.use_bn = use_bn
        self.use_ds = use_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Down Sample
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.conv00 = UnetConv2d(self.in_channel, filters[0], self.use_bn)
        self.conv10 = UnetConv2d(filters[0], filters[1], self.use_bn)
        self.conv20 = UnetConv2d(filters[1], filters[2], self.use_bn)
        self.conv30 = UnetConv2d(filters[2], filters[3], self.use_bn)
        self.conv40 = UnetConv2d(filters[3], filters[4], self.use_bn)

        # Up Sample
        self.up_concat01 = UnetUp(filters[1], filters[0], self.use_deconv, 2)
        self.up_concat11 = UnetUp(filters[2], filters[1], self.use_deconv, 2)
        self.up_concat21 = UnetUp(filters[3], filters[2], self.use_deconv, 2)
        self.up_concat31 = UnetUp(filters[4], filters[3], self.use_deconv, 2)

        self.up_concat02 = UnetUp(filters[1], filters[0], self.use_deconv, 3)
        self.up_concat12 = UnetUp(filters[2], filters[1], self.use_deconv, 3)
        self.up_concat22 = UnetUp(filters[3], filters[2], self.use_deconv, 3)

        self.up_concat03 = UnetUp(filters[1], filters[0], self.use_deconv, 4)
        self.up_concat13 = UnetUp(filters[2], filters[1], self.use_deconv, 4)

        self.up_concat04 = UnetUp(filters[1], filters[0], self.use_deconv, 5)

        # Finale Convolution
        self.final1 = nn.Conv2d(filters[0], n_class, 1)
        self.final2 = nn.Conv2d(filters[0], n_class, 1)
        self.final3 = nn.Conv2d(filters[0], n_class, 1)
        self.final4 = nn.Conv2d(filters[0], n_class, 1)
        self.stack = P.Stack(axis=0)

    def construct(self, inputs):
        x00 = self.conv00(inputs)                   # channel = filters[0]
        x10 = self.conv10(self.maxpool(x00))        # channel = filters[1]
        x20 = self.conv20(self.maxpool(x10))        # channel = filters[2]
        x30 = self.conv30(self.maxpool(x20))        # channel = filters[3]
        x40 = self.conv40(self.maxpool(x30))        # channel = filters[4]

        x01 = self.up_concat01(x10, x00)            # channel = filters[0]
        x11 = self.up_concat11(x20, x10)            # channel = filters[1]
        x21 = self.up_concat21(x30, x20)            # channel = filters[2]
        x31 = self.up_concat31(x40, x30)            # channel = filters[3]

        x02 = self.up_concat02(x11, x00, x01)       # channel = filters[0]
        x12 = self.up_concat12(x21, x10, x11)       # channel = filters[1]
        x22 = self.up_concat22(x31, x20, x21)       # channel = filters[2]

        x03 = self.up_concat03(x12, x00, x01, x02)  # channel = filters[0]
        x13 = self.up_concat13(x22, x10, x11, x12)  # channel = filters[1]

        x04 = self.up_concat04(x13, x00, x01, x02, x03) # channel = filters[0]

        final1 = self.final1(x01)
        final2 = self.final2(x02)
        final3 = self.final3(x03)
        final4 = self.final4(x04)

        if self.use_ds:
            final = self.stack((final1, final2, final3, final4))
            return final
        return final4


class UNet(nn.Cell):
    """
    Simple UNet with skip connection
    """
    def __init__(self, in_channel, n_class=2, feature_scale=2, use_deconv=True, use_bn=True):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.n_class = n_class
        self.feature_scale = feature_scale
        self.use_deconv = use_deconv
        self.use_bn = use_bn

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Down Sample
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.conv0 = UnetConv2d(self.in_channel, filters[0], self.use_bn)
        self.conv1 = UnetConv2d(filters[0], filters[1], self.use_bn)
        self.conv2 = UnetConv2d(filters[1], filters[2], self.use_bn)
        self.conv3 = UnetConv2d(filters[2], filters[3], self.use_bn)
        self.conv4 = UnetConv2d(filters[3], filters[4], self.use_bn)

        # Up Sample
        self.up_concat1 = UnetUp(filters[1], filters[0], self.use_deconv, 2)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.use_deconv, 2)
        self.up_concat3 = UnetUp(filters[3], filters[2], self.use_deconv, 2)
        self.up_concat4 = UnetUp(filters[4], filters[3], self.use_deconv, 2)

        # Finale Convolution
        self.final = nn.Conv2d(filters[0], n_class, 1)

    def construct(self, inputs):
        x0 = self.conv0(inputs)                   # channel = filters[0]
        x1 = self.conv1(self.maxpool(x0))        # channel = filters[1]
        x2 = self.conv2(self.maxpool(x1))        # channel = filters[2]
        x3 = self.conv3(self.maxpool(x2))        # channel = filters[3]
        x4 = self.conv4(self.maxpool(x3))        # channel = filters[4]

        up4 = self.up_concat4(x4, x3)
        up3 = self.up_concat3(up4, x2)
        up2 = self.up_concat2(up3, x1)
        up1 = self.up_concat1(up2, x0)

        final = self.final(up1)

        return final

if __name__ == "__main__":
    # pynative模式
    context.set_context(mode=context.PYNATIVE_MODE, pynative_synchronize=True)
#list: {Tensor[Float16], Tensor[Float32], Tensor[Int32], Tensor[Int64], Tensor[Int8]}.
    backbone = UNet(16)
    x = ms.Tensor(np.random.randint(0, 10, [1, 16, 1024, 512]), ms.float32)
    y_ms = backbone(x)
    print(y_ms)
    
""" $lic$
Copyright (C) 2016-2020 by Tsinghua University and The Board of Trustees of
Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer

'''
VGGNet-16

Simonyan and Zisserman, 2014
'''

NN = Network('VGG')

NN.set_input_layer(InputLayer(3, 76))

NN.add('conv1', ConvLayer(3, 64, 78, 3))
NN.add('conv2', ConvLayer(64, 64, 78, 3))
NN.add('pool1', PoolingLayer(64, 43, 2))

NN.add('conv3', ConvLayer(64, 128, 45, 3))
NN.add('conv4', ConvLayer(128, 128, 44, 3))
NN.add('pool2', PoolingLayer(128, 26, 2))

NN.add('conv5', ConvLayer(128, 256, 28, 3))
NN.add('conv6', ConvLayer(256, 256, 30, 3))
NN.add('conv7', ConvLayer(256, 256, 30, 3))
NN.add('pool3', PoolingLayer(256, 19, 2))

NN.add('conv8', ConvLayer(256, 512, 21, 3))
NN.add('conv9', ConvLayer(512, 512, 23, 3))
NN.add('conv10', ConvLayer(512, 512, 22, 3))
NN.add('pool4', PoolingLayer(512, 15, 2))

NN.add('conv11', ConvLayer(512, 512, 15, 3))
NN.add('conv12', ConvLayer(512, 512, 15, 3))
NN.add('conv13', ConvLayer(512, 512, 14, 3))
NN.add('pool5', PoolingLayer(512, 7, 2))

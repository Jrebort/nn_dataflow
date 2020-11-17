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
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, \
        PoolingLayer, EltwiseLayer

'''
ResNet-50

He, Zhang, Ren, and Sun, 2015
'''

NN = Network('ResNet')

NN.set_input_layer(InputLayer(3, 52))

NN.add('conv1_a', ConvLayer(3, 64, 52, 1, 2))
NN.add('conv1_b', ConvLayer(64, 64, 54, 3, 2)) 

for i in range(3):
    NN.add('conv2_{}_a'.format(i), ConvLayer(64 if i == 0 else 256, 64, 54, 1))
    if i == 0:
        NN.add('conv2_{}_b'.format(i), ConvLayer(64, 64, 28, 3, 2))
    else:
        NN.add('conv2_{}_b'.format(i), ConvLayer(64, 64, 28, 1))
    NN.add('conv2_{}_c'.format(i), ConvLayer(64, 256, 28, 1))

    # With residual shortcut.
    if i == 0:
        NN.add('conv2_{}_res'.format(i), ConvLayer(256, 256, 27, 1, 2))
    NN.add('conv2_{}_d'.format(i), ConvLayer(256, 256, 28, 1))

for i in range(4):
    NN.add('conv3_{}_a'.format(i), ConvLayer(256, 128, 28, 1, 1) if i == 0
           else ConvLayer(512, 128, 56, 1))
    if i == 0:
        NN.add('conv3_{}_b'.format(i), ConvLayer(128, 128, 15, 3, 2))
    else:
        NN.add('conv3_{}_b'.format(i), ConvLayer(128, 128, 15, 1))
    NN.add('conv3_{}_c'.format(i), ConvLayer(128, 512, 15, 1))

    if i == 0:
        NN.add('conv3_{}_res'.format(i), ConvLayer(512, 512, 14, 1, 2))
    NN.add('conv3_{}_d'.format(i), ConvLayer(512, 512, 15, 1))

for i in range(6):
    NN.add('conv4_{}_a'.format(i), ConvLayer(512, 256, 15, 1, 1) if i == 0
           else ConvLayer(1024, 256, 56, 1))
    if i == 0:
        NN.add('conv4_{}_b'.format(i), ConvLayer(256, 256, 9, 3, 2))
    else:
        NN.add('conv4_{}_b'.format(i), ConvLayer(256, 256, 9, 1))
    NN.add('conv4_{}_c'.format(i), ConvLayer(256, 1024, 9, 1))

    if i == 0:
        NN.add('conv4_{}_res'.format(i), ConvLayer(1024, 1024, 9, 1, 2))
    NN.add('conv4_{}_d'.format(i), ConvLayer(1024, 1024, 9, 1))


for i in range(3):
    NN.add('conv5_{}_a'.format(i),
           ConvLayer(1024, 512, 14, 1) if i == 0
           else ConvLayer(2048, 512, 14, 1))
    NN.add('conv5_{}_b'.format(i), ConvLayer(512, 512, 14, 1))
    NN.add('conv5_{}_c'.format(i), ConvLayer(512, 2048, 14, 1))

    if i == 0:
        NN.add('conv5_{}_res'.format(i), ConvLayer(2048, 2048, 14, 1))
    NN.add('conv5_{}_d'.format(i), ConvLayer(2048, 2048, 14, 1))

NN.add('pool5', PoolingLayer(2048, 1, 7))

NN.add('fc', FCLayer(2048, 1000))

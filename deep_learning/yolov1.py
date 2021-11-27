import torch
import torch.nn as nn

# output_size = [(input_size - kernel_size + 2 * padding) / stride] + 1

# from yolov1 paper, architecture (kernel_size, filters, stride, padding):
# convolutional layers  only

model_config = [
    # block 1
    (7, 64, 2, 3),
    "maxpool2_2",

    # block 2
    (3, 192, 1,  1),
    "maxpool2_2",

    # block 3
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "maxpool2_2",

    # block 4
    # conv (1, 256, 1, 0)
    # conv (3, 512, 1, 1)
    # above 2 layers * 4
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "maxpool2_2",

    # block 5
    # conv (1, 512, 1, 0)
    # conv (3, 1024, 1, 1)
    # above 2 layers * 2
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),

    # block 6 
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()

        # batch norm is added, though batch norm was not implemented in original paper

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.leaky_relu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()

        self.architecture = model_config
        self.in_channels = in_channels

        self.conv_layers = self._create_conv_layers(self.architecture)
        self.fully_conn = self._create_fully_conn(**kwargs)

    def forward(self, x):
        x = self.conv_layers(x)

        return self.fully_conn(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []

        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, tuple):
                layers += [
                    CNNBlock(
                        in_channels,
                        out_channels=x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3]
                    )
                ]

                in_channels = x[1]

            elif x == "maxpool2_2":
                layers += [
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2
                    )
                ]
            elif isinstance(x, list):
                conv1 = x[0]
                conv2 = x[1]
                n_reps = x[2]

                for _ in range(n_reps):
                    layers += [
                        CNNBlock(
                            in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    ]

                    layers += [
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fully_conn(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)) # output shape = (S, S, 30) as B = 2, C = 20
        )

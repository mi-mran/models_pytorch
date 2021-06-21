import torch
import torch.nn as nn

# this model uses convtranspose2d instead of upsample

def convolutions_consecutive(input_c, output_c):
    layers = nn.Sequential(
        nn.Conv2d(input_c, output_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_c, output_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )

    return layers

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class UNet (nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_conv_1 = convolutions_consecutive(self.in_channels, 64)
        self.down_conv_2 = convolutions_consecutive(64, 128)
        self.down_conv_3 = convolutions_consecutive(128, 256)
        self.down_conv_4 = convolutions_consecutive(256, 512)
        self.down_conv_5 = convolutions_consecutive(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            1024, 
            512,
            kernel_size=2,
            stride=2
            )

        self.up_conv_1 = convolutions_consecutive(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(
            512, 
            256,
            kernel_size=2,
            stride=2
            )

        self.up_conv_2 = convolutions_consecutive(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            256, 
            128,
            kernel_size=2,
            stride=2
            )

        self.up_conv_3 = convolutions_consecutive(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(
            128, 
            64,
            kernel_size=2,
            stride=2
            )

        self.up_conv_4 = convolutions_consecutive(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=self.out_channels,
            kernel_size=1
        )

    def forward(self, input_img):
        
        #encoding layers

        x1 = self.down_conv_1(input_img) # for concatenation
        print(f'x1: {x1.size()}')
        x1a = self.max_pool_2x2(x1)

        x2 = self.down_conv_2(x1a) # for concatenation
        print(f'x2: {x2.size()}')
        x2a = self.max_pool_2x2(x2)

        x3 = self.down_conv_3(x2a) # for concatenation
        print(f'x3: {x3.size()}')
        x3a = self.max_pool_2x2(x3)

        x4 = self.down_conv_4(x3a) # for concatenation
        print(f'x4: {x4.size()}')
        x4a = self.max_pool_2x2(x4)

        x5 = self.down_conv_5(x4a)
        print(f'x5: {x5.size()}')

        #decoding layers

        x6 = self.up_trans_1(x5)
        print(f'x6: {x6.size()}')
        y6 = crop_img(x4, x6)
        print(f'y6: {y6.size()}')
        x6a = self.up_conv_1(torch.cat([x6, y6], dim=1))
        print(f'x6a: {x6a.size()}')

        x7 = self.up_trans_2(x6a)
        print(f'x7: {x7.size()}')
        y7 = crop_img(x3, x7)
        print(f'y7: {y7.size()}')
        x7a = self.up_conv_2(torch.cat([x7, y7], dim=1))
        print(f'x7a: {x7a.size()}')

        x8 = self.up_trans_3(x7a)
        print(f'x8: {x8.size()}')
        y8 = crop_img(x2, x8)
        print(f'y8: {y8.size()}')
        x8a = self.up_conv_3(torch.cat([x8, y8], dim=1))
        print(f'x8a: {x8a.size()}')

        x9 = self.up_trans_4(x8a)
        print(f'x9: {x9.size()}')
        y9 = crop_img(x1, x9)
        print(f'y9: {y9.size()}')
        x9a = self.up_conv_4(torch.cat([x9, y9], dim=1))
        print(f'x9a: {x9a.size()}')
        
        x10 = self.out(x9a)
        print(f'x10: {x10.size()}')

        return x10

if __name__ == '__main__':
    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    print(model(image).size())
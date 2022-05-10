######################################## Double Convolution
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

    
######################################## Maxpooling followed by Double Convolution
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


######################################## Upsampling followed by Double Convolution
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0),
        ) 
        self.conv = DoubleConv(out_channels * 2, out_channels)


    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

######################################## Output layer (1x1 Convolution followed by SoftMax activation)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv_sigmoid = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv_sigmoid(x)

class RfR_model(nn.Module):

    #inter_in_channels = 64
    #last_in_channels = 32 + 3 # 32(context) + 3(flow)
    
    def __init__(self, name, in_channels, out_channels):
        super(RfR_model, self).__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.inputL = DoubleConv(self.in_channels, self.in_channels)
        self.down1 = Down(self.in_channels, 2*self.in_channels)
        self.down2 = Down(2*self.in_channels, 2*self.in_channels)
        self.down3 = Down(2*self.in_channels, 4*self.in_channels)
        self.down4 = Down(4*self.in_channels, 4*self.in_channels)
        self.down5 = Down(4*self.in_channels, 8*self.in_channels)
        self.down6 = Down(8*self.in_channels, 8*self.in_channels)
        
        self.up1 = Up(8*self.in_channels, 8*self.in_channels)
        self.up2 = Up(8*self.in_channels, 4*self.in_channels)
        self.up3 = Up(4*self.in_channels, 4*self.in_channels)
        self.up4 = Up(4*self.in_channels, 2*self.in_channels)
        self.up5 = Up(2*self.in_channels, 2*self.in_channels)
        self.up6 = Up(2*self.in_channels, self.in_channels)
        self.outputL = OutConv(self.in_channels, self.out_channels)
        
    def forward(self, x):
        x1 = self.inputL(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        b = self.down6(x6)
        
        x = self.up1(b, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        
        x = self.outputL(x)
        
        return x
        
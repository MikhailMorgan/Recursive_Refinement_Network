class RfR(nn.Module):

    inter_in_channels = 64
    last_in_channels = 32 + 3 # 32(context) + 3(flow)
    
    def __init__(self, name, n_channels, n_classes):
        super(RfR, self).__init__()
        self.name = name
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inputL = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 256)
        self.down6 = Down(256, 512)
        self.down7 = Down(512, 512)
        self.down8 = Down(512, 1024)
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 256)
        self.up5 = Up(256, 128)
        self.up6 = Up(128, 128)
        self.up7 = Up(128, 64)
        self.up8 = Up(64, 64)
        self.outputL = OutConv(64, 3)
        
    def forward(self, x):
        x1 = self.inputL(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        b = self.down8(x8)
        
        x = self.up1(b, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)
        
        x = self.outputL(x)
        
        return x
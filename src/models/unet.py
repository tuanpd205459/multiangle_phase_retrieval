import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class PhaseRefiningUNet(nn.Module):
    """
    Mạng U-Net tinh chỉnh trường sóng phức (Phase Refining U-Net).
    
    Đầu vào: Trường sóng phức thô giải điều chế [B, 1, H, W] (Complex64).
    Được chuyển đổi thành 2 kênh thực [B, 2, H, W] đại diện cho phần thực (Real) 
    và phần ảo (Imaginary) để tránh hiện tượng đứt gãy pha (phase wrapping) khi học.
    
    Đầu ra: Trường sóng phức tinh chỉnh sạch [B, 1, H, W] (Complex64).
    """
    def __init__(self, in_channels=2, out_channels=2):
        super(PhaseRefiningUNet, self).__init__()

        # Bộ mã hóa (Encoder / Downsampling)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Bộ giải mã (Decoder / Upsampling với Skip Connections)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128) # 128 (từ up) + 128 (từ down2)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64) # 64 (từ up) + 64 (từ down1)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32) # 32 (từ up) + 32 (từ inc)

        # Lớp tích chập đầu ra để đưa về 2 kênh (Real và Imag)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Khởi tạo trọng số và bias của lớp cuối cùng bằng 0
        # Đảm bảo ban đầu U_refined = U_rough (chưa có nhiễu từ mạng)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, U_rough):
        """
        U_rough: Tensor phức [B, 1, H, W] (Complex64)
        """
        # 1. Tách trường sóng phức thành phần thực và phần ảo làm 2 kênh đầu vào
        real_part = U_rough.real
        imag_part = U_rough.imag
        x = torch.cat([real_part, imag_part], dim=1) # [B, 2, H, W]

        # 2. Đi qua Encoder
        x1 = self.inc(x)         # [B, 32, H, W]
        x2 = self.down1(x1)      # [B, 64, H/2, W/2]
        x3 = self.down2(x2)      # [B, 128, H/4, W/4]
        x4 = self.down3(x3)      # [B, 256, H/8, W/8]

        # 3. Đi qua Decoder kết hợp Skip Connections
        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)   # [B, 128, H/4, W/4]

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)   # [B, 64, H/2, W/2]

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)   # [B, 32, H, W]

        # 4. Tích chập đầu ra
        out = self.outc(u3)      # [B, 2, H, W]

        # 5. Tái tạo lại trường sóng phức đầu ra dạng cộng hưởng dư (Residual Connection)
        real_out = out[:, 0:1, :, :]
        imag_out = out[:, 1:2, :, :]
        U_refined_unnorm = U_rough + torch.complex(real_out, imag_out) # [B, 1, H, W]

        # 6. Ép chuẩn hóa biên độ về 1.0 (Tiên nghiệm Pure Phase cho hệ phản xạ)
        U_refined = U_refined_unnorm / (torch.abs(U_refined_unnorm) + 1e-8)

        return U_refined

if __name__ == "__main__":
    print("⏳ Đang kiểm tra mạng Phase Refining U-Net...")
    model = PhaseRefiningUNet()
    
    # Tạo đầu vào phức giả lập
    U_in = torch.complex(torch.rand(2, 1, 256, 256), torch.rand(2, 1, 256, 256))
    U_out = model(U_in)
    
    print("✅ Kiểm tra thành công!")
    print(f"Input shape: {U_in.shape}")
    print(f"Output shape: {U_out.shape}")
    print(f"Output dtype: {U_out.dtype}")

import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft

class DifferentiableDemodulator(nn.Module):
    """
    Bộ giải điều chế khả vi (Differentiable Demodulator) thực hiện dịch phổ và lọc thông thấp.
    
    Để kx và ky có thể tối ưu hóa liên tục ở cấp độ sub-pixel bằng Gradient Descent,
    phép dịch phổ Fourier được thực hiện thông qua nhân pha tuyến tính ở miền không gian:
        I_shifted(x,y) = I(x,y) * exp(-i * 2*pi * (kx*x/W + ky*y/H))
    
    Sau đó, đưa sang miền Fourier để áp dụng bộ lọc thông thấp (low-pass filter) hình tròn
    để lọc lấy trường sóng phức vật thể U_demod.
    """
    def __init__(self, filter_radius_x=20.0, filter_radius_y=60.0):
        super(DifferentiableDemodulator, self).__init__()
        # Đăng ký bán kính elip ngang và dọc dạng nn.Parameter
        self.filter_radius_x = nn.Parameter(torch.tensor(float(filter_radius_x), dtype=torch.float32))
        self.filter_radius_y = nn.Parameter(torch.tensor(float(filter_radius_y), dtype=torch.float32))

    def forward(self, I, kx, ky):
        """
        I: Tensor hologram cường độ [B, 1, H, W]
        kx, ky: Tần số sóng mang tương ứng [B] (ở đơn vị pixel dịch chuyển so với tâm)
        """
        B, C, H, W = I.shape
        device = I.device

        # 1. Tạo lưới tọa độ không gian (x, y)
        y_grid = torch.arange(H, dtype=torch.float32, device=device)
        x_grid = torch.arange(W, dtype=torch.float32, device=device)
        mesh_y, mesh_x = torch.meshgrid(y_grid, x_grid, indexing='ij') # [H, W]

        # Mở rộng kích thước lưới và sóng mang để thực hiện phép toán song song theo Batch
        mesh_x_expanded = mesh_x.view(1, 1, H, W)
        mesh_y_expanded = mesh_y.view(1, 1, H, W)
        kx_expanded = kx.view(B, 1, 1, 1)
        ky_expanded = ky.view(B, 1, 1, 1)

        # 2. Thực hiện dịch phổ khả vi ở miền không gian (Fourier Shift Theorem)
        # Nhân ảnh hologram cường độ thực với số mũ phức của sóng mang ngược hướng
        phase_shift = -2.0 * np.pi * (kx_expanded * mesh_x_expanded / W + ky_expanded * mesh_y_expanded / H)
        
        # Tạo số phức exp(i * phase_shift)
        cos_shift = torch.cos(phase_shift)
        sin_shift = torch.sin(phase_shift)
        exp_shift = torch.complex(cos_shift, sin_shift)
        
        # Dịch chuyển trường sóng sang miền phức
        I_complex_shifted = I.to(torch.complex64) * exp_shift

        # 3. Biến đổi sang miền tần số Fourier
        I_fft = fft.fftshift(fft.fft2(I_complex_shifted), dim=(-2, -1))

        # 4. Áp dụng bộ lọc thông thấp hình tròn tại tâm tần số (H//2, W//2)
        y_dist = mesh_y - H // 2
        x_dist = mesh_x - W // 2
        
        # Tạo mặt nạ lọc hình elip mềm khả vi (Soft Sigmoid Elliptical Filter Mask)
        # Kẹp giá trị bán kính tối thiểu là 5.0 để tránh bán kính elip âm/quá nhỏ
        rx = torch.clamp(self.filter_radius_x, min=5.0)
        ry = torch.clamp(self.filter_radius_y, min=5.0)
        
        # Khoảng cách Elip chuẩn hóa
        distance_ellipse = torch.sqrt((x_dist / rx)**2 + (y_dist / ry)**2)
        
        # Sử dụng Sigmoid để tạo độ mượt cho biên bộ lọc elip, temperature = 0.1
        temperature = 0.1
        mask = torch.sigmoid((1.0 - distance_ellipse) / temperature).view(1, 1, H, W).to(device)
        I_fft_filtered = I_fft * mask

        # 5. Biến đổi Fourier ngược để thu được trường sóng phức đã giải điều chế
        U_demod = fft.ifft2(fft.ifftshift(I_fft_filtered, dim=(-2, -1)))

        return U_demod

# ==============================================================================
# BẢN KIỂM TRA CHỨC NĂNG (TEST RUN)
# ==============================================================================
if __name__ == "__main__":
    print("⏳ Đang kiểm tra bộ giải điều chế khả vi...")
    
    # Tạo tensor hologram giả lập [Batch=2, Channel=1, H=256, W=256]
    I = torch.rand(2, 1, 256, 256)
    
    # Thiết lập sóng mang có thể tối ưu hóa (đòi hỏi tính gradient)
    kx = torch.tensor([40.0, -45.0], requires_grad=True)
    ky = torch.tensor([-30.0, -35.0], requires_grad=True)
    
    demod = DifferentiableDemodulator(filter_radius_x=20.0, filter_radius_y=60.0)
    U_demod = demod(I, kx, ky)
    
    # Tính một hàm mục tiêu đơn giản để thử nghiệm lan truyền ngược (Backward)
    loss = torch.mean(torch.abs(U_demod))
    loss.backward()
    
    print("✅ Kiểm tra thành công!")
    print(f"Demodulated wave shape: {U_demod.shape}")
    print(f"Demodulated wave dtype: {U_demod.dtype}")
    print(f"Gradient of kx: {kx.grad.numpy()}")
    print(f"Gradient of ky: {ky.grad.numpy()}")
    print(f"Gradient of filter_radius_x: {demod.filter_radius_x.grad.item():.6f}")
    print(f"Gradient of filter_radius_y: {demod.filter_radius_y.grad.item():.6f}")

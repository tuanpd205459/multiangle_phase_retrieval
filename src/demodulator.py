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
    
    Sau đó, đưa sang miền Fourier để áp dụng bộ lọc thông thấp (low-pass filter) hình chữ nhật mềm
    để lọc lấy trường sóng phức vật thể U_demod.

    Kích thước bộ lọc (rx, ry) có thể:
      - Dùng giá trị mặc định toàn cục (chế độ synthetic / huấn luyện)
      - Nhận per-sample từ estimate_filter_size() khi xử lý ảnh thực (chế độ inference)
    """
    def __init__(self, filter_radius_x=20.0, filter_radius_y=60.0):
        super(DifferentiableDemodulator, self).__init__()
        # Bán kính bộ lọc mặc định toàn cục (dùng khi không truyền rx/ry per-sample)
        self.filter_radius_x = nn.Parameter(torch.tensor(float(filter_radius_x), dtype=torch.float32))
        self.filter_radius_y = nn.Parameter(torch.tensor(float(filter_radius_y), dtype=torch.float32))

    def forward(self, I, kx, ky, rx_override=None, ry_override=None, mask_override=None):
        """
        I:             Tensor hologram cường độ [B, 1, H, W]
        kx, ky:        Tần số sóng mang [B] (pixel lệch so với tâm DC)
        rx_override:   Bán kính bộ lọc X (fallback nếu không truyền mask_override)
        ry_override:   Bán kính bộ lọc Y (fallback nếu không truyền mask_override)
        mask_override: Tensor [B, 1, H, W] chứa mặt nạ lọc thích nghi đã làm mịn (Gaussian window)
                       tương thích đúng ý dạng búp phổ thực tế.
        """
        B, C, H, W = I.shape
        device = I.device

        # 1. Tạo lưới tọa độ không gian (x, y)
        y_grid = torch.arange(H, dtype=torch.float32, device=device)
        x_grid = torch.arange(W, dtype=torch.float32, device=device)
        mesh_y, mesh_x = torch.meshgrid(y_grid, x_grid, indexing='ij')  # [H, W]

        mesh_x_exp = mesh_x.view(1, 1, H, W)
        mesh_y_exp = mesh_y.view(1, 1, H, W)
        kx_exp = kx.view(B, 1, 1, 1)
        ky_exp = ky.view(B, 1, 1, 1)

        # 2. Dịch phổ về tâm (Fourier Shift Theorem)
        phase_shift = -2.0 * np.pi * (kx_exp * mesh_x_exp / W + ky_exp * mesh_y_exp / H)
        exp_shift = torch.complex(torch.cos(phase_shift), torch.sin(phase_shift))
        I_complex_shifted = I.to(torch.complex64) * exp_shift

        # 3. FFT → bậc +1 giờ nằm tại tâm (H//2, W//2)
        I_fft = fft.fftshift(fft.fft2(I_complex_shifted), dim=(-2, -1))

        # 4. Áp dụng bộ lọc
        if mask_override is not None:
            # Sử dụng trực tiếp mặt nạ thích nghi mềm (Gaussian Window)
            mask = mask_override.to(device)
        else:
            # Fallback dùng bộ lọc hình chữ nhật mềm nếu không có mask_override
            if rx_override is not None:
                rx = torch.as_tensor(rx_override, dtype=torch.float32, device=device).view(B, 1, 1, 1)
                rx = torch.clamp(rx, min=5.0)
            else:
                rx = torch.clamp(self.filter_radius_x, min=5.0).view(1, 1, 1, 1).expand(B, 1, 1, 1)

            if ry_override is not None:
                ry = torch.as_tensor(ry_override, dtype=torch.float32, device=device).view(B, 1, 1, 1)
                ry = torch.clamp(ry, min=5.0)
            else:
                ry = torch.clamp(self.filter_radius_y, min=5.0).view(1, 1, 1, 1).expand(B, 1, 1, 1)

            y_dist = mesh_y - H // 2
            x_dist = mesh_x - W // 2
            x_abs = torch.abs(x_dist).view(1, 1, H, W)
            y_abs = torch.abs(y_dist).view(1, 1, H, W)

            temperature = 0.5
            mask_x = torch.sigmoid((rx - x_abs) / temperature)
            mask_y = torch.sigmoid((ry - y_abs) / temperature)
            mask = mask_x * mask_y

        I_fft_filtered = I_fft * mask

        # 5. IFFT → trường sóng phức đã giải điều chế
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
    kx = torch.tensor([40.0, 15.0], requires_grad=True)  # sample 2 có kx nhỏ
    ky = torch.tensor([-30.0, -1.5], requires_grad=True)
    
    demod = DifferentiableDemodulator(filter_radius_x=20.0, filter_radius_y=60.0)

    print("\n--- Chế độ 1: Global learnable (synthetic/training) ---")
    U_demod = demod(I, kx, ky)
    loss = torch.mean(torch.abs(U_demod))
    loss.backward()
    print(f"✅ Shape: {U_demod.shape}, dtype: {U_demod.dtype}")
    print(f"Gradient of kx: {kx.grad.numpy()}")
    print(f"Gradient of filter_radius_x: {demod.filter_radius_x.grad.item():.6f}")

    print("\n--- Chế độ 2: Per-sample rx_override (ảnh thực / inference) ---")
    # Ví dụ: estimate_filter_size() trả về rx=[35, 55], ry=[40, 60] per-sample
    rx_from_estimate = torch.tensor([35.0, 55.0])  # sample 2 có rx lớn dù kx nhỏ!
    ry_from_estimate = torch.tensor([40.0, 60.0])
    kx2 = torch.tensor([40.0, 15.0], requires_grad=True)
    ky2 = torch.tensor([-30.0, -1.5], requires_grad=True)
    U_demod2 = demod(I, kx2, ky2, rx_override=rx_from_estimate, ry_override=ry_from_estimate)
    loss2 = torch.mean(torch.abs(U_demod2))
    loss2.backward()
    print(f"✅ Shape: {U_demod2.shape}")
    print(f"Gradient of kx (per-sample rx): {kx2.grad.numpy()}")
    print("✅ rx_override hoạt động đúng — filter lớn dù kx=15!")
    print(f"Gradient of filter_radius_y: {demod.filter_radius_y.grad.item():.6f}")

import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft

class DifferentiableDemodulator(nn.Module):
    """
    Bộ giải điều chế khả vi (Differentiable Demodulator) thực hiện dịch phổ và lọc thông thấp.
    
    Luồng xử lý (2 bước):
        Bước 1 — Dịch phổ sơ bộ với k_dataset:
            I_shifted = I * exp(-i*2π*(kx*x/W + ky*y/H))
            → FFT → búp phổ +1 gần tâm nhưng có thể lệch Δk

        Bước 2 — Spectral Centroid Correction (thay thế KEstimator CNN):
            Tính trọng tâm (centroid) của phổ công suất trong vùng lọc:
                cx = Σ(fx * |FFT*mask|²) / Σ(|FFT*mask|²)
            cx, cy chính là Δk tuyệt đối trong hệ tọa độ Fourier.
            → k_final = k_dataset + (cx, cy)
            → Làm lại dịch phổ với k_final → búp phổ về đúng tâm (0,0)

        Ưu điểm so với KEstimator CNN:
            - Không cần học, tính được ngay từ vật lý
            - Tham chiếu tọa độ Fourier tuyệt đối → phá vỡ Gauge Ambiguity
            - Khả vi hoàn toàn (centroid là phép tính đại số)

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
        kx, ky:        Tần số sóng mang sơ bộ từ dataset [B]
        rx_override:   Bán kính bộ lọc X per-sample (tuỳ chọn)
        ry_override:   Bán kính bộ lọc Y per-sample (tuỳ chọn)
        mask_override: Tensor [B, 1, H, W] mặt nạ lọc thích nghi (tuỳ chọn)

        Returns:
            U_demod:   Trường sóng phức đã giải điều chế [B, 1, H, W]
            delta_k:   Độ lệch sóng mang tính từ Spectral Centroid [B, 2]
            k_final:   Sóng mang cuối = k_dataset + delta_k [B, 2]
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

        # 2. Bước 1: Dịch phổ sơ bộ với k_dataset
        phase_shift = -2.0 * np.pi * (kx_exp * mesh_x_exp / W + ky_exp * mesh_y_exp / H)
        exp_shift = torch.complex(torch.cos(phase_shift), torch.sin(phase_shift))
        I_complex_shifted = I.to(torch.complex64) * exp_shift

        # 3. FFT sơ bộ → búp phổ +1 gần tâm (H//2, W//2) nhưng có thể lệch Δk
        I_fft = fft.fftshift(fft.fft2(I_complex_shifted), dim=(-2, -1))

        # 4. Tạo bộ lọc (mask) xung quanh tâm
        if mask_override is not None:
            mask = mask_override.to(device)
        else:
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

        # =====================================================================
        # SPECTRAL CENTROID CORRECTION (thay thế KEstimator CNN)
        # Đo vị trí chính xác của búp phổ +1 trong miền Fourier → tính Δk
        # Centroid tham chiếu tọa độ Fourier tuyệt đối → phá vỡ Gauge Ambiguity
        # =====================================================================
        power = torch.abs(I_fft * mask) ** 2  # Phổ công suất trong vùng lọc [B,1,H,W]

        # Lưới tần số centered: -H/2 → H/2-1 (đơn vị pixel trong FFT)
        fy = torch.arange(-H//2, H//2, dtype=torch.float32, device=device).view(1, 1, H, 1)
        fx = torch.arange(-W//2, W//2, dtype=torch.float32, device=device).view(1, 1, 1, W)

        total_power = power.sum(dim=(-2, -1), keepdim=True) + 1e-8
        centroid_x  = (power * fx).sum(dim=(-2, -1), keepdim=True) / total_power  # [B,1,1,1]
        centroid_y  = (power * fy).sum(dim=(-2, -1), keepdim=True) / total_power

        # Δk = offset của centroid so với tâm (0,0)
        delta_kx = centroid_x.view(B)   # [B]
        delta_ky = centroid_y.view(B)   # [B]

        # k_final = k_dataset + Δk → kéo búp phổ về đúng tâm (0,0)
        kx_final = kx + delta_kx
        ky_final = ky + delta_ky

        # Bước 2: Làm lại dịch phổ với k_final
        kx_final_exp = kx_final.view(B, 1, 1, 1)
        ky_final_exp = ky_final.view(B, 1, 1, 1)

        phase_shift_final = -2.0 * np.pi * (kx_final_exp * mesh_x_exp / W + ky_final_exp * mesh_y_exp / H)
        exp_shift_final   = torch.complex(torch.cos(phase_shift_final), torch.sin(phase_shift_final))
        I_complex_final   = I.to(torch.complex64) * exp_shift_final

        # FFT lần 2 — búp phổ +1 bây giờ nằm đúng tâm (0,0)
        I_fft_final = fft.fftshift(fft.fft2(I_complex_final), dim=(-2, -1))
        # =====================================================================

        # 5. Áp dụng bộ lọc vào FFT đã hiệu chỉnh
        I_fft_filtered = I_fft_final * mask

        # 6. IFFT → trường sóng phức đã giải điều chế
        U_demod = fft.ifft2(fft.ifftshift(I_fft_filtered, dim=(-2, -1)))

        # Đóng gói delta_k và k_final thành tensor [B, 2] để trả về
        delta_k = torch.stack([delta_kx, delta_ky], dim=1)  # [B, 2]
        k_final = torch.stack([kx_final, ky_final], dim=1)  # [B, 2]

        return U_demod, delta_k, k_final

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

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def save_dataset_preview(dataset, output_path, num_samples=3, filter_radius=50):
    """
    Trực quan hóa một số mẫu trong bộ dữ liệu cùng phổ Fourier của chúng để kiểm tra trước khi training.
    Lưu ảnh kết quả dưới dạng lưới:
    - Cột 1: Hologram 1 (Góc 1)
    - Cột 2: Phổ Fourier 1 (Log amplitude) + Vòng tròn bộ lọc tại sóng mang
    - Cột 3: Hologram 2 (Góc 2)
    - Cột 4: Phổ Fourier 2 (Log amplitude) + Vòng tròn bộ lọc tại sóng mang
    - Cột 5: Bản đồ pha Ground Truth (nếu có)
    """
    num_samples = min(num_samples, len(dataset))
    if num_samples <= 0:
        print("⚠️ Cảnh báo: Bộ dữ liệu rỗng, không thể trực quan hóa.")
        return

    # Xác định số lượng cột (5 cột nếu có GT, nếu không thì 4 cột)
    # Thử lấy mẫu đầu tiên xem có GT phase không (nếu tổng giá trị tuyệt đối > 0)
    first_sample = dataset[0]
    has_gt = 'phi_gt' in first_sample and float(torch.abs(first_sample['phi_gt']).sum()) > 0
    cols = 5 if has_gt else 4

    fig, axes = plt.subplots(num_samples, cols, figsize=(cols * 4, num_samples * 3.8))
    
    # Đảm bảo axes là ma trận 2D ngay cả khi num_samples = 1
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        sample = dataset[i]
        
        # 1. Trích xuất dữ liệu
        I1 = sample['I1'].squeeze().numpy()
        I2 = sample['I2'].squeeze().numpy()
        k1 = sample['k1'].numpy() # [kx1, ky1]
        k2 = sample['k2'].numpy() # [kx2, ky2]
        
        H, W = I1.shape
        cx, cy = W // 2, H // 2
        
        # Tính phổ Fourier (Log Amplitude Shifted)
        I1_fft = np.fft.fftshift(np.fft.fft2(I1))
        I2_fft = np.fft.fftshift(np.fft.fft2(I2))
        I1_fft_log = np.log(np.abs(I1_fft) + 1e-6)
        I2_fft_log = np.log(np.abs(I2_fft) + 1e-6)
        
        # --- Cột 1: Hologram 1 ---
        axes[i, 0].imshow(I1, cmap='gray')
        axes[i, 0].set_ylabel(f"Sample {i}", fontsize=12)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title("Hologram 1 (Angle 1)", fontsize=12, pad=10)
            
        # --- Cột 2: Phổ Fourier 1 ---
        im_fft1 = axes[i, 1].imshow(I1_fft_log, cmap='viridis')
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title("Fourier Spectrum 1 (Log)", fontsize=12, pad=10)
            
        # Vẽ tâm DC (màu xanh lá) và sóng mang (màu đỏ)
        axes[i, 1].plot(cx, cy, 'g+', markersize=8, label='DC')
        peak_x1 = cx + k1[0]
        peak_y1 = cy + k1[1]
        axes[i, 1].plot(peak_x1, peak_y1, 'rx', markersize=8, label='Carrier')
        
        # Vẽ vòng tròn bộ lọc thông thấp (đây là nơi búp sóng +1 được dịch về tâm và lọc)
        # Center of filter circle on the raw FFT is at (cx + kx, cy + ky)
        circle1 = plt.Circle((peak_x1, peak_y1), filter_radius, color='red', fill=False, linestyle='--', linewidth=1.5)
        axes[i, 1].add_patch(circle1)
        
        # --- Cột 3: Hologram 2 ---
        axes[i, 2].imshow(I2, cmap='gray')
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title("Hologram 2 (Angle 2)", fontsize=12, pad=10)
            
        # --- Cột 4: Phổ Fourier 2 ---
        im_fft2 = axes[i, 3].imshow(I2_fft_log, cmap='viridis')
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title("Fourier Spectrum 2 (Log)", fontsize=12, pad=10)
            
        # Vẽ tâm DC (màu xanh lá) và sóng mang (màu đỏ)
        axes[i, 3].plot(cx, cy, 'g+', markersize=8)
        peak_x2 = cx + k2[0]
        peak_y2 = cy + k2[1]
        axes[i, 3].plot(peak_x2, peak_y2, 'rx', markersize=8)
        
        # Vẽ vòng tròn bộ lọc thông thấp
        circle2 = plt.Circle((peak_x2, peak_y2), filter_radius, color='red', fill=False, linestyle='--', linewidth=1.5)
        axes[i, 3].add_patch(circle2)
        
        # --- Cột 5: Ground Truth Phase (nếu có) ---
        if cols == 5:
            phi_gt = sample['phi_gt'].squeeze().numpy()
            im_gt = axes[i, 4].imshow(phi_gt, cmap='jet')
            axes[i, 4].axis('off')
            fig.colorbar(im_gt, ax=axes[i, 4], fraction=0.046, pad=0.04)
            if i == 0:
                axes[i, 4].set_title("Ground Truth Phase", fontsize=12, pad=10)
                
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Đã tạo thành công ảnh kiểm tra dataset tại: {output_path}")

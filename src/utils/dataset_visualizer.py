import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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
            
        # In tọa độ sóng mang của mẫu ra Console
        print(f"   - Mẫu {i}: k1=[{k1[0]:.2f}, {k1[1]:.2f}] | k2=[{k2[0]:.2f}, {k2[1]:.2f}]")

        # --- Cột 2: Phổ Fourier 1 ---
        im_fft1 = axes[i, 1].imshow(I1_fft_log, cmap='viridis')
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title(f"Fourier 1 (Log)\nk1=({k1[0]:.2f}, {k1[1]:.2f})", fontsize=12, pad=10)
        else:
            axes[i, 1].set_title(f"k1=({k1[0]:.2f}, {k1[1]:.2f})", fontsize=10)
            
        # Vẽ tâm DC (màu xanh lá) và sóng mang (màu đỏ)
        axes[i, 1].plot(cx, cy, 'g+', markersize=8, label='DC')
        peak_x1 = cx + k1[0]
        peak_y1 = cy + k1[1]
        axes[i, 1].plot(peak_x1, peak_y1, 'rx', markersize=8, label='Carrier')
        
        # Vẽ mặt nạ elip bộ lọc thông thấp (Rx = 0.4 * R, Ry = 1.2 * R)
        # Giới hạn Rx tương tự mô hình để không chạm vào vệt sáng DC (x = cx)
        rx1 = filter_radius * 0.4
        max_rx1 = abs(k1[0]) * 0.8
        rx1 = min(rx1, max_rx1)
        ry1 = filter_radius * 1.2
        ellipse1 = Ellipse((peak_x1, peak_y1), width=2*rx1, height=2*ry1, angle=0, color='red', fill=False, linestyle='--', linewidth=1.5)
        axes[i, 1].add_patch(ellipse1)
        
        # --- Cột 3: Hologram 2 ---
        axes[i, 2].imshow(I2, cmap='gray')
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title("Hologram 2 (Angle 2)", fontsize=12, pad=10)
            
        # --- Cột 4: Phổ Fourier 2 ---
        im_fft2 = axes[i, 3].imshow(I2_fft_log, cmap='viridis')
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title(f"Fourier 2 (Log)\nk2=({k2[0]:.2f}, {k2[1]:.2f})", fontsize=12, pad=10)
        else:
            axes[i, 3].set_title(f"k2=({k2[0]:.2f}, {k2[1]:.2f})", fontsize=10)
            
        # Vẽ tâm DC (màu xanh lá) và sóng mang (màu đỏ)
        axes[i, 3].plot(cx, cy, 'g+', markersize=8)
        peak_x2 = cx + k2[0]
        peak_y2 = cy + k2[1]
        axes[i, 3].plot(peak_x2, peak_y2, 'rx', markersize=8)
        
        # Vẽ mặt nạ elip bộ lọc thông thấp
        rx2 = filter_radius * 0.4
        max_rx2 = abs(k2[0]) * 0.8
        rx2 = min(rx2, max_rx2)
        ry2 = filter_radius * 1.2
        ellipse2 = Ellipse((peak_x2, peak_y2), width=2*rx2, height=2*ry2, angle=0, color='red', fill=False, linestyle='--', linewidth=1.5)
        axes[i, 3].add_patch(ellipse2)
        
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

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import torch
import sys

# Import estimate_filter_size để tính kích thước bộ lọc đúng từ phổ thực tế
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.dataset import estimate_filter_size


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
        im_fft1 = axes[i, 1].imshow(I1_fft_log, cmap='viridis', vmin=0, vmax=12)
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
        
        # Vẽ mặt nạ búp phổ thích nghi mềm dạng đường bao (Contour) màu đỏ
        rx1, ry1, mask1_centered, _ = estimate_filter_size(I1, k1[0], k1[1])
        # Dịch chuyển mask về đúng tọa độ của búp phổ để vẽ
        mask1_orig = np.roll(mask1_centered, int(round(k1[1])), axis=0)
        mask1_orig = np.roll(mask1_orig, int(round(k1[0])), axis=1)
        axes[i, 1].contour(mask1_orig, levels=[0.25], colors='red', linewidths=1.2, linestyles='--')
        
        # --- Cột 3: Hologram 2 ---
        axes[i, 2].imshow(I2, cmap='gray')
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title("Hologram 2 (Angle 2)", fontsize=12, pad=10)
            
        # --- Cột 4: Phổ Fourier 2 ---
        im_fft2 = axes[i, 3].imshow(I2_fft_log, cmap='viridis', vmin=0, vmax=12)
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
        
        # Vẽ mặt nạ búp phổ thích nghi mềm dạng đường bao (Contour) màu đỏ cho góc 2
        rx2, ry2, mask2_centered, _ = estimate_filter_size(I2, k2[0], k2[1])
        mask2_orig = np.roll(mask2_centered, int(round(k2[1])), axis=0)
        mask2_orig = np.roll(mask2_orig, int(round(k2[0])), axis=1)
        axes[i, 3].contour(mask2_orig, levels=[0.25], colors='red', linewidths=1.2, linestyles='--')
        
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

def save_intermediate_steps_preview(dataset, output_path, sample_idx=0, filter_radius=50):
    """
    Trực quan hóa chi tiết các bước trung gian của quá trình lọc thích nghi
    và giải điều chế vật lý 2D FFT cho cả 2 góc chiếu của một mẫu vật.
    Hiển thị đúng:
      - Cột 1: Hologram gốc
      - Cột 2: Phổ Fourier gốc + Bounding Box đỏ
      - Cột 3: Mặt nạ nhị phân thích nghi của búp phổ +1 đã chọn (sau khi dọn DC)
      - Cột 4: Cửa sổ lọc mềm (Gaussian Window)
      - Cột 5: Pha thô tái tạo (Wrapped Phase)
    """
    import cv2
    import scipy.ndimage as ndimage
    
    if len(dataset) <= sample_idx:
        print(f"⚠️ Cảnh báo: sample_idx {sample_idx} vượt quá kích thước dataset.")
        return
        
    sample = dataset[sample_idx]
    I1 = sample['I1'].squeeze().numpy()
    I2 = sample['I2'].squeeze().numpy()
    k1 = sample['k1'].numpy() # [kx1, ky1]
    k2 = sample['k2'].numpy() # [kx2, ky2]
    
    H, W = I1.shape
    cx, cy = W // 2, H // 2
    
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    
    angles_data = [
        {'I': I1, 'k': k1, 'title_suffix': 'Angle 1'},
        {'I': I2, 'k': k2, 'title_suffix': 'Angle 2'}
    ]
    
    for i, data in enumerate(angles_data):
        I = data['I']
        k = data['k']
        suffix = data['title_suffix']
        
        # 1. Phổ Fourier gốc (Log Amplitude)
        I_fft_raw = np.fft.fftshift(np.fft.fft2(I))
        I_fft_raw_log = np.log(np.abs(I_fft_raw) + 1e-6)
        
        # 2. Gọi trực tiếp hàm estimate_filter_size đã được đồng bộ chuẩn xác ở dataset.py
        rx, ry, mask_centered, binary_mask = estimate_filter_size(I, k[0], k[1])
        
        # Bounding Box thực tế từ binary_mask
        target_x = cx + k[0]
        target_y = cy + k[1]
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_contour = None
        bx, by, bw_w, bw_h = int(target_x) - 15, int(target_y) - 15, 30, 30
        
        if len(contours) > 0:
            best_dist = float('inf')
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    c_x = M["m10"] / M["m00"]
                    c_y = M["m01"] / M["m00"]
                    dist = np.sqrt((c_x - target_x)**2 + (c_y - target_y)**2)
                    if dist < best_dist:
                        best_dist = dist
                        target_contour = cnt
            if target_contour is not None:
                bx, by, bw_w, bw_h = cv2.boundingRect(target_contour)
                
        # 3. Tạo bộ lọc mềm Gaussian từ mặt nạ nhị phân thực tế (sigma=8)
        filter_window = ndimage.gaussian_filter(binary_mask, sigma=8.0)
        filter_window = filter_window / (filter_window.max() + 1e-8)
        
        # 4. Phục dựng pha thô
        # Dịch phổ về tâm (shift)
        y_grid = np.arange(H)
        x_grid = np.arange(W)
        mesh_y, mesh_x = np.meshgrid(y_grid, x_grid, indexing='ij')
        exp_shift = np.exp(-2j * np.pi * (k[0] * mesh_x / W + k[1] * mesh_y / H))
        I_shifted = I.astype(np.complex64) * exp_shift
        I_fft_shifted = np.fft.fftshift(np.fft.fft2(I_shifted))
        
        # Tạo cửa sổ mềm dịch tâm từ mask_centered
        I_fft_filtered = I_fft_shifted * mask_centered
        U_rough = np.fft.ifft2(np.fft.ifftshift(I_fft_filtered))
        phase_rough = np.angle(U_rough)
        
        # --- Cột 1: Hologram gốc ---
        axes[i, 0].imshow(I, cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Input Hologram ({suffix})", fontsize=11)
        
        # --- Cột 2: Phổ Fourier gốc (kèm Bounding Box đỏ) ---
        axes[i, 1].imshow(I_fft_raw_log, cmap='viridis', vmin=0, vmax=12)
        axes[i, 1].axis('off')
        axes[i, 1].plot(cx, cy, 'g+', markersize=8) # DC center
        axes[i, 1].plot(target_x, target_y, 'rx', markersize=8) # Carrier center
        if target_contour is not None:
            rect = Rectangle((bx, by), width=bw_w, height=bw_h, color='red', fill=False, linestyle='--', linewidth=1.5)
            axes[i, 1].add_patch(rect)
        axes[i, 1].set_title(f"Fourier + BBox\nk=({k[0]:.2f}, {k[1]:.2f})", fontsize=10)
        
        # --- Cột 3: Mặt nạ nhị phân của búp phổ ---
        axes[i, 2].imshow(binary_mask, cmap='gray')
        axes[i, 2].axis('off')
        axes[i, 2].set_title("Selected Lobe Mask", fontsize=10)
        
        # --- Cột 4: Gaussian filter window ---
        axes[i, 3].imshow(filter_window, cmap='jet')
        axes[i, 3].axis('off')
        axes[i, 3].set_title("Gaussian Soft Window", fontsize=10)
        
        # --- Cột 5: Pha thô giải điều chế ---
        im_phase = axes[i, 4].imshow(phase_rough, cmap='jet', vmin=-np.pi, vmax=np.pi)
        axes[i, 4].axis('off')
        axes[i, 4].set_title("Demodulated Phase", fontsize=10)
        fig.colorbar(im_phase, ax=axes[i, 4], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Đã tạo thành công ảnh kiểm tra bước trung gian tại: {output_path}")

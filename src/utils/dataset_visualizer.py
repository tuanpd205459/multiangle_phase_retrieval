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
        rx1, ry1, mask1_centered = estimate_filter_size(I1, k1[0], k1[1])
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
        rx2, ry2, mask2_centered = estimate_filter_size(I2, k2[0], k2[1])
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
        
        # 2. Chạy thuật toán tìm 3 đối tượng và lấy thông tin vùng bậc +1
        norm_spec = (spec_smooth - spec_smooth.min()) / (spec_smooth.max() - spec_smooth.min() + 1e-8)
        norm_spec_u8 = (norm_spec * 255).astype(np.uint8)
        gtl, _ = cv2.threshold(norm_spec_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gtl = gtl / 255.0
        
        T = gtl
        step = 0.01 * gtl
        max_iter = 100
        best_bw = None
        num_labels_prev = 0
        
        # Kernel cho morphology
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        for _ in range(max_iter):
            bw = (spec_smooth >= T).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bw_area = np.zeros_like(bw)
            for cnt in contours:
                if cv2.contourArea(cnt) > 30:
                    cv2.drawContours(bw_area, [cnt], -1, 255, -1)
            bw_close = cv2.morphologyEx(bw_area, cv2.MORPH_CLOSE, kernel_close)
            contours_fill, _ = cv2.findContours(bw_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bw_filled = np.zeros_like(bw_close)
            cv2.drawContours(bw_filled, contours_fill, -1, 255, -1)
            bw_open = cv2.morphologyEx(bw_filled, cv2.MORPH_OPEN, kernel_open)
            
            num_labels, _ = cv2.connectedComponents(bw_open)
            num_objects = num_labels - 1
            
            if num_objects == 3:
                best_bw = bw_open
                break
            elif num_objects == 2 and (best_bw is None or num_labels_prev != 3):
                best_bw = bw_open
                
            num_labels_prev = num_objects
            T += step
            if T >= 1.0:
                break
                
        if best_bw is None:
            best_bw = bw_open
            
        contours_final, _ = cv2.findContours(best_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_x = cx + k[0]
        target_y = cy + k[1]
        best_dist = float('inf')
        target_contour = None
        
        for cnt in contours_final:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                c_x = M["m10"] / M["m00"]
                c_y = M["m01"] / M["m00"]
                dist = np.sqrt((c_x - target_x)**2 + (c_y - target_y)**2)
                if dist < best_dist:
                    best_dist = dist
                    target_contour = cnt
                    
        # Tạo mặt nạ thích nghi thực tế dựa trên Convex Hull
        binary_mask = np.zeros((H, W), dtype=np.float32)
        bx, by, bw_w, bw_h = 0, 0, 15, 15
        if target_contour is not None:
            hull = cv2.convexHull(target_contour)
            cv2.drawContours(binary_mask, [hull], -1, 1.0, -1)
            bx, by, bw_w, bw_h = cv2.boundingRect(hull)
        else:
            cv2.circle(binary_mask, (int(target_x), int(target_y)), 15, 1.0, -1)
            bx, by = int(target_x) - 7, int(target_y) - 7
            
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
        
        # Dịch chuyển mặt nạ nhị phân về tâm DC (roll theo -kx, -ky)
        shift_x = int(round(-k[0]))
        shift_y = int(round(-k[1]))
        binary_mask_c = np.roll(binary_mask, shift_y, axis=0)
        binary_mask_c = np.roll(binary_mask_c, shift_x, axis=1)
        
        # Tạo cửa sổ mềm dịch tâm
        filter_window_c = ndimage.gaussian_filter(binary_mask_c, sigma=8.0)
        filter_window_c = filter_window_c / (filter_window_c.max() + 1e-8)
        
        # Nhân phổ dịch tâm với bộ lọc mềm
        I_fft_filtered = I_fft_shifted * filter_window_c
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
        if target_prop is not None:
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

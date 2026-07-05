import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
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
        
        # Vẽ mặt nạ hình chữ nhật bộ lọc thông thấp (Rx = 0.4 * R, Ry = 1.2 * R)
        # Giới hạn Rx tương tự mô hình để không chạm vào vệt sáng DC (x = cx)
        rx1 = filter_radius * 0.4
        max_rx1 = abs(k1[0]) * 0.8
        rx1 = min(rx1, max_rx1)
        ry1 = filter_radius * 1.2
        rect1 = Rectangle((peak_x1 - rx1, peak_y1 - ry1), width=2*rx1, height=2*ry1, color='red', fill=False, linestyle='--', linewidth=1.5)
        axes[i, 1].add_patch(rect1)
        
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
        
        # Vẽ mặt nạ hình chữ nhật bộ lọc thông thấp
        rx2 = filter_radius * 0.4
        max_rx2 = abs(k2[0]) * 0.8
        rx2 = min(rx2, max_rx2)
        ry2 = filter_radius * 1.2
        rect2 = Rectangle((peak_x2 - rx2, peak_y2 - ry2), width=2*rx2, height=2*ry2, color='red', fill=False, linestyle='--', linewidth=1.5)
        axes[i, 3].add_patch(rect2)
        
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
    Trực quan hóa chi tiết các bước trung gian của quá trình giải điều chế vật lý 2D FFT
    (trước khi đưa vào U-Net) cho cả 2 góc chiếu của một mẫu vật.
    Giúp người dùng kiểm chứng pha thô và biên độ thô (điểm tựa baseline).
    """
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
    y_grid = np.arange(H)
    x_grid = np.arange(W)
    mesh_y, mesh_x = np.meshgrid(y_grid, x_grid, indexing='ij')
    
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    
    # Danh sách dữ liệu cho 2 góc
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
        
        # 2. Nhân dịch tần số vật lý: I_shifted = I * exp(-2i*pi*(kx*x/W + ky*y/H))
        exp_shift = np.exp(-2j * np.pi * (k[0] * mesh_x / W + k[1] * mesh_y / H))
        I_shifted = I.astype(np.complex64) * exp_shift
        
        # 3. Biến đổi sang miền tần số sau khi dịch chuyển
        I_fft_shifted = np.fft.fftshift(np.fft.fft2(I_shifted))
        
        # 4. Áp dụng bộ lọc hình chữ nhật
        rx = filter_radius * 0.4
        max_rx = abs(k[0]) * 0.8
        rx = min(rx, max_rx)
        ry = filter_radius * 1.2
        
        x_dist = mesh_x - W // 2
        y_dist = mesh_y - H // 2
        
        # Soft rectangular mask along x and y dimensions (using sigmoids)
        mask_x = 1.0 / (1.0 + np.exp(-(rx - np.abs(x_dist)) / 0.1))
        mask_y = 1.0 / (1.0 + np.exp(-(ry - np.abs(y_dist)) / 0.1))
        mask = mask_x * mask_y
        
        I_fft_filtered = I_fft_shifted * mask
        I_fft_filtered_log = np.log(np.abs(I_fft_filtered) + 1e-6)
        
        # 5. Biến đổi Fourier ngược thu được U_rough (trường phức thô)
        U_rough = np.fft.ifft2(np.fft.ifftshift(I_fft_filtered))
        amp_rough = np.abs(U_rough)
        phase_rough = np.angle(U_rough)
        
        # --- Cột 1: Hologram gốc ---
        axes[i, 0].imshow(I, cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Input Hologram ({suffix})", fontsize=11)
        
        # --- Cột 2: Phổ Fourier gốc (kèm HCN đỏ) ---
        axes[i, 1].imshow(I_fft_raw_log, cmap='viridis', vmin=0, vmax=12)
        axes[i, 1].axis('off')
        axes[i, 1].plot(cx, cy, 'g+', markersize=8) # DC center
        peak_x = cx + k[0]
        peak_y = cy + k[1]
        axes[i, 1].plot(peak_x, peak_y, 'rx', markersize=8) # Carrier center
        
        rect = Rectangle((peak_x - rx, peak_y - ry), width=2*rx, height=2*ry, color='red', fill=False, linestyle='--', linewidth=1.5)
        axes[i, 1].add_patch(rect)
        axes[i, 1].set_title(f"Fourier + Rect filter\nk=({k[0]:.2f}, {k[1]:.2f})", fontsize=10)
        
        # --- Cột 3: Phổ sau khi lọc và dịch về tâm DC ---
        axes[i, 2].imshow(I_fft_filtered_log, cmap='viridis', vmin=0, vmax=12)
        axes[i, 2].axis('off')
        axes[i, 2].set_title("Filtered & Shifted Fourier", fontsize=10)
        
        # --- Cột 4: Biên độ thô giải điều chế ---
        im_amp = axes[i, 3].imshow(amp_rough, cmap='jet')
        axes[i, 3].axis('off')
        axes[i, 3].set_title("Raw Amplitude (Baseline)", fontsize=10)
        fig.colorbar(im_amp, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        # --- Cột 5: Pha thô giải điều chế (Wrapped) ---
        im_phase = axes[i, 4].imshow(phase_rough, cmap='jet', vmin=-np.pi, vmax=np.pi)
        axes[i, 4].axis('off')
        axes[i, 4].set_title("Raw Wrapped Phase (Baseline)", fontsize=10)
        fig.colorbar(im_phase, ax=axes[i, 4], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Đã tạo thành công ảnh kiểm tra bước trung gian (Fourier demodulation) tại: {output_path}")

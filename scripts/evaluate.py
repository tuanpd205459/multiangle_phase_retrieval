import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt

# Thêm thư mục gốc của dự án vào sys.path để chạy từ bất kỳ đâu không bị lỗi import src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Nạp các module từ thư mục src
from src.dataset import MultiAngleHologramDataset
from src.models.siamese import SiameseTeacherModel

def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Khôi phục Pha Tự Giám Sát Đa Góc")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Đường dẫn đến file cấu hình YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Đường dẫn đến file checkpoint (.pth) tốt nhất")
    parser.add_argument("--num_test", type=int, default=5,
                        help="Số lượng mẫu thực hiện trực quan hóa")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_psnr(img_gt, img_pred):
    """Tính chỉ số PSNR giữa hai ma trận ảnh"""
    mse = np.mean((img_gt - img_pred) ** 2)
    if mse == 0:
        return float('inf')
    # Cho pha wrapped, dải giá trị tối đa là 2*pi
    max_val = 2 * np.pi
    return 20 * np.log10(max_val / np.sqrt(mse))

def align_phase_wrapped(phase_gt_wrapped, phase_pred_wrapped):
    """
    Bù lệch pha toàn cục cho pha quấn (wrapped phase) trước khi tính sai số.
    """
    U_gt = np.exp(1j * phase_gt_wrapped)
    U_pred = np.exp(1j * phase_pred_wrapped)
    
    # Ước lượng độ lệch pha toàn cục bằng tích vô hướng phức
    delta_psi = np.angle(np.sum(U_gt * np.conj(U_pred)))
    
    # Bù pha
    phase_pred_aligned = phase_pred_wrapped + delta_psi
    
    # Đưa pha về khoảng [-pi, pi]
    return np.angle(np.exp(1j * phase_pred_aligned))

def evaluate():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device(config['cloud']['device'] if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Sử dụng thiết bị: {device} để chạy đánh giá (Chế độ so sánh pha Wrapped).")
        
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Khởi tạo Dataset kiểm thử
    dataset_mode = 'synthetic' if config['data']['synthetic_data'] else 'real'
    print(f"📦 Đang nạp dữ liệu kiểm thử ở chế độ: {dataset_mode.upper()}")
    
    test_dataset = MultiAngleHologramDataset(
        mode=dataset_mode,
        data_dir=config['data']['raw_dir'] if dataset_mode == 'real' else None,
        num_samples=100 if dataset_mode == 'synthetic' else 50,
        image_size=(config['data']['image_height'], config['data']['image_width']),
        seed=100,
        is_eval=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. Nạp mô hình và trọng số
    model = SiameseTeacherModel(filter_radius=config['data']['filter_radius']).to(device)
    if os.path.exists(args.checkpoint):
        print(f"🔄 Đang nạp trọng số mô hình từ: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"⚠️ Không tìm thấy file checkpoint tại {args.checkpoint}. Chạy suy diễn bằng trọng số ngẫu nhiên.")
        
    model.eval()
    
    # 3. Chạy suy diễn và tính toán định lượng
    mse_list = []
    psnr_list = []
    
    print("⏳ Đang tiến hành suy diễn...")
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            I1 = batch['I1'].to(device)
            I2 = batch['I2'].to(device)
            k1 = batch['k1'].to(device)
            k2 = batch['k2'].to(device)
            
            # Khôi phục trường sóng phức
            (U1, amp1, phase1), (U2, amp2, phase2) = model(I1, k1, I2, k2)
            
            # Chuyển dữ liệu về numpy
            phi_pred = phase1[0, 0].cpu().numpy()
            amp_pred = amp1[0, 0].cpu().numpy()
            
            # Xuất ma trận kết quả sang MATLAB (.mat) cho mẫu đầu tiên
            if idx == 0:
                mat_path = os.path.join(output_dir, "reconstructed_sample.mat")
                mat_data = {
                    'hologram1': I1[0, 0].cpu().numpy(),
                    'hologram2': I2[0, 0].cpu().numpy(),
                    'amplitude1': amp_pred,
                    'phase1': phi_pred,
                    'kx1': k1[0, 0].item(),
                    'ky1': k1[0, 1].item()
                }
                
                # Nếu là dữ liệu giả lập, lưu thêm pha nhãn Ground Truth (được quấn về [-pi, pi])
                if dataset_mode == 'synthetic':
                    phi_gt_unwrapped = batch['phi_gt'][0, 0].cpu().numpy()
                    phi_gt_wrapped = np.angle(np.exp(1j * phi_gt_unwrapped))
                    mat_data['phase_gt'] = phi_gt_wrapped
                    
                sio.savemat(mat_path, mat_data)
                print(f"💾 Đã lưu dữ liệu ma trận khôi phục mẫu đầu tiên vào: {mat_path} (Hỗ trợ MATLAB)")
            
            # Tính chỉ số định lượng nếu chạy trên dữ liệu giả lập (có Ground Truth)
            if dataset_mode == 'synthetic':
                phi_gt_unwrapped = batch['phi_gt'][0, 0].cpu().numpy()
                phi_gt_wrapped = np.angle(np.exp(1j * phi_gt_unwrapped))
                
                # Căn chỉnh pha quấn dự đoán với pha quấn Ground Truth
                phi_pred_aligned = align_phase_wrapped(phi_gt_wrapped, phi_pred)
                
                mse = np.mean((phi_gt_wrapped - phi_pred_aligned) ** 2)
                psnr = compute_psnr(phi_gt_wrapped, phi_pred_aligned)
                
                mse_list.append(mse)
                psnr_list.append(psnr)
            else:
                phi_gt_wrapped = None
                phi_pred_aligned = phi_pred
                
            # Trực quan hóa hình ảnh
            if idx < args.num_test:
                visualize_sample(
                    idx,
                    I1[0, 0].cpu().numpy(),
                    I2[0, 0].cpu().numpy(),
                    amp_pred,
                    phi_pred_aligned,
                    phi_gt_wrapped,
                    output_dir
                )
                
    if dataset_mode == 'synthetic' and len(mse_list) > 0:
        print(f"📊 Kết quả đánh giá định lượng trên toàn bộ tập Test (Pha Wrapped):")
        print(f"   - Phase MSE Trung bình: {np.mean(mse_list):.6f}")
        print(f"   - Phase PSNR Trung bình: {np.mean(psnr_list):.2f} dB")
        
    print(f"🎨 Đã xuất các hình ảnh trực quan hóa vào thư mục: {output_dir}")

def visualize_sample(sample_idx, I1, I2, amp, phase, phase_gt, output_dir):
    """
    Vẽ bảng so sánh kết quả 2x3 hiển thị các bản đồ pha wrapped.
    """
    cols = 4 if phase_gt is not None else 3
    fig, axes = plt.subplots(2, cols, figsize=(15, 7))
    
    # ---------------- DÒNG 1: KẾT QUẢ KHÔI PHỤC ----------------
    # 1. Hologram góc 1
    axes[0, 0].imshow(I1, cmap='gray')
    axes[0, 0].set_title("Input Hologram 1")
    axes[0, 0].axis('off')
    
    # 2. Biên độ khôi phục
    im_amp = axes[0, 1].imshow(amp, cmap='jet')
    axes[0, 1].set_title("Reconstructed Amplitude")
    axes[0, 1].axis('off')
    fig.colorbar(im_amp, ax=axes[0, 1])
    
    # 3. Pha quấn khôi phục (Reconstructed Wrapped Phase)
    im_phase = axes[0, 2].imshow(phase, cmap='jet', vmin=-np.pi, vmax=np.pi)
    axes[0, 2].set_title("Reconstructed Phase (Wrapped)")
    axes[0, 2].axis('off')
    fig.colorbar(im_phase, ax=axes[0, 2])
    
    if phase_gt is not None:
        # 4. Pha Ground Truth Quấn (Ground Truth Wrapped Phase)
        im_gt = axes[0, 3].imshow(phase_gt, cmap='jet', vmin=-np.pi, vmax=np.pi)
        axes[0, 3].set_title("Ground Truth Phase (Wrapped)")
        axes[0, 3].axis('off')
        fig.colorbar(im_gt, ax=axes[0, 3])
        
    # ---------------- DÒNG 2: ĐỐI CHỨNG VÀ ĐÁNH GIÁ ----------------
    # 5. Hologram góc 2
    axes[1, 0].imshow(I2, cmap='gray')
    axes[1, 0].set_title("Input Hologram 2")
    axes[1, 0].axis('off')
    
    # 6. Phổ Fourier của Hologram 1
    I1_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(I1))) + 1e-3)
    axes[1, 1].imshow(I1_fft, cmap='viridis')
    axes[1, 1].set_title("Hologram Fourier Spectrum")
    axes[1, 1].axis('off')
    
    if phase_gt is not None:
        # 7. Bản đồ lỗi pha quấn (Phase Error Map)
        # Để đo lỗi pha quấn chính xác, ta lấy độ lệch pha phức để tránh bước nhảy 2pi
        error_map = np.abs(np.angle(np.exp(1j * (phase_gt - phase))))
        im_err = axes[1, 2].imshow(error_map, cmap='hot', vmin=0, vmax=np.pi)
        axes[1, 2].set_title("Wrapped Phase Error Map")
        axes[1, 2].axis('off')
        fig.colorbar(im_err, ax=axes[1, 2])
        
        # 8. Đồ thị Profile cắt ngang của pha quấn
        mid_row = phase_gt.shape[0] // 2
        axes[1, 3].plot(phase_gt[mid_row, :], 'k-', label='Ground Truth')
        axes[1, 3].plot(phase[mid_row, :], 'r--', label='Reconstructed')
        axes[1, 3].set_title("Phase Mid-line Profile")
        axes[1, 3].legend()
        axes[1, 3].grid(True)
    else:
        # Đối với dữ liệu thực tế (Không có Ground Truth)
        # 7. Đồ thị Profile cắt ngang của pha quấn
        mid_row = phase.shape[0] // 2
        axes[1, 2].plot(phase[mid_row, :], 'b-', label='Phase Profile')
        axes[1, 2].set_title("Wrapped Phase Mid-line Profile")
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # 8. Tắt trục còn lại
        axes[1, 3].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"visual_evaluation_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    evaluate()

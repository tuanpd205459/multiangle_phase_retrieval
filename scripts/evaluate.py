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
            
            mask1 = batch.get('mask1', None)
            mask2 = batch.get('mask2', None)
            if mask1 is not None:
                mask1 = mask1.to(device)
            if mask2 is not None:
                mask2 = mask2.to(device)
                
            # Khôi phục trường sóng phức
            (U1, amp1, phase1, phase_rough1, k1_final, delta_k1), \
            (U2, amp2, phase2, phase_rough2, k2_final, delta_k2) = model(I1, k1, I2, k2, mask1=mask1, mask2=mask2)
            
            # Tính toán Hologram tái tạo (Reconstructed Hologram) để kiểm chứng vật lý
            B, C, H, W = U1.shape
            y_grid = torch.arange(H, dtype=torch.float32, device=device)
            x_grid = torch.arange(W, dtype=torch.float32, device=device)
            mesh_y, mesh_x = torch.meshgrid(y_grid, x_grid, indexing='ij')
            mesh_x_expanded = mesh_x.view(1, 1, H, W)
            mesh_y_expanded = mesh_y.view(1, 1, H, W)
            # Tính hologram cường độ dự đoán cho GÓC 1 và cùng thức cho GÓC 2
            kx1 = k1_final[0, 0].view(1, 1, 1, 1)
            ky1 = k1_final[0, 1].view(1, 1, 1, 1)
            phase_carrier1 = 2.0 * np.pi * (kx1 * mesh_x_expanded / W + ky1 * mesh_y_expanded / H)
            R1 = torch.complex(torch.cos(phase_carrier1), torch.sin(phase_carrier1))
            I_pred1 = torch.abs(U1 + R1) ** 2
            scale1 = torch.mean(I1, dim=(-2,-1), keepdim=True) / (torch.mean(I_pred1, dim=(-2,-1), keepdim=True) + 1e-8)
            I_pred1_np = (I_pred1 * scale1)[0, 0].cpu().numpy()

            # Hologram tái tạo GÓC 2 (dùng k2_final và U2)
            kx2 = k2_final[0, 0].view(1, 1, 1, 1)
            ky2 = k2_final[0, 1].view(1, 1, 1, 1)
            phase_carrier2 = 2.0 * np.pi * (kx2 * mesh_x_expanded / W + ky2 * mesh_y_expanded / H)
            R2 = torch.complex(torch.cos(phase_carrier2), torch.sin(phase_carrier2))
            I_pred2 = torch.abs(U2 + R2) ** 2
            scale2 = torch.mean(I2, dim=(-2,-1), keepdim=True) / (torch.mean(I_pred2, dim=(-2,-1), keepdim=True) + 1e-8)
            I_pred2_np = (I_pred2 * scale2)[0, 0].cpu().numpy()
            
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
                    'phase_rough1': phase_rough1[0, 0].cpu().numpy(),
                    'phase_rough2': phase_rough2[0, 0].cpu().numpy(),
                    # K thô từ dataset
                    'kx1_dataset': k1[0, 0].item(),
                    'ky1_dataset': k1[0, 1].item(),
                    # K cuối cùng sau khi cộng Δk học được
                    'kx1_final': k1_final[0, 0].item(),
                    'ky1_final': k1_final[0, 1].item(),
                    # Δk học được
                    'delta_kx1': delta_k1[0, 0].item(),
                    'delta_ky1': delta_k1[0, 1].item(),
                    'delta_kx2': delta_k2[0, 0].item(),
                    'delta_ky2': delta_k2[0, 1].item(),
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
                    I_pred1_np,
                    I_pred2_np,
                    amp_pred,
                    phi_pred_aligned,
                    phase_rough1[0, 0].cpu().numpy(),
                    phase2[0, 0].cpu().numpy(),
                    phase_rough2[0, 0].cpu().numpy(),
                    phi_gt_wrapped,
                    output_dir
                )
                
    if dataset_mode == 'synthetic' and len(mse_list) > 0:
        print(f"📊 Kết quả đánh giá định lượng trên toàn bộ tập Test (Pha Wrapped):")
        print(f"   - Phase MSE Trung bình: {np.mean(mse_list):.6f}")
        print(f"   - Phase PSNR Trung bình: {np.mean(psnr_list):.2f} dB")
        
    print(f"🎨 Đã xuất các hình ảnh trực quan hóa vào thư mục: {output_dir}")

def visualize_sample(sample_idx, I1, I2, I_pred1, I_pred2,
                     amp, phase1, phase_rough1,
                     phase2, phase_rough2,
                     phase_gt, output_dir):
    """
    Vẽ bảng so sánh kết quả đầy đủ 3 dòng:
      Dòng 1: Kết quả GÓC 1  — Hologram gốc | Hologram tái tạo | Biên độ | Pha tinh chỉnh | Pha trung gian | Pha Delta
      Dòng 2: Kết quả GÓC 2  — Hologram gốc | Hologram tái tạo | Phổ Fourier | Pha tinh chỉnh | Pha trung gian | Pha Delta
      Dòng 3: Đánh giá      — Profile so sánh | Ground Truth (nếu có) | Error Map (nếu có)
    """
    ncols = 6
    fig, axes = plt.subplots(3, ncols, figsize=(ncols * 3.8, 11))

    def _off(ax): ax.axis('off')
    def _cbar(ax, im): fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ======== DÒNG 1: GÓC 1 ========
    axes[0, 0].imshow(I1, cmap='gray')
    axes[0, 0].set_title("Input Hologram 1"); _off(axes[0, 0])

    axes[0, 1].imshow(I_pred1, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title("Reconstructed Hologram 1"); _off(axes[0, 1])

    im = axes[0, 2].imshow(amp, cmap='jet')
    axes[0, 2].set_title("Reconstructed Amplitude"); _off(axes[0, 2]); _cbar(axes[0, 2], im)

    im = axes[0, 3].imshow(phase1, cmap='jet', vmin=-np.pi, vmax=np.pi)
    axes[0, 3].set_title("Refined Phase 1 (Wrapped)"); _off(axes[0, 3]); _cbar(axes[0, 3], im)

    im = axes[0, 4].imshow(phase_rough1, cmap='jet', vmin=-np.pi, vmax=np.pi)
    axes[0, 4].set_title("Intermediate Phase 1 (Wrapped)"); _off(axes[0, 4]); _cbar(axes[0, 4], im)

    # Delta = pha tinh chỉnh - pha trung gian (sự khác biệt do U-Net thêm vào)
    delta1 = np.angle(np.exp(1j * (phase1 - phase_rough1)))
    im = axes[0, 5].imshow(delta1, cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    axes[0, 5].set_title("Phase Refinement Δφ₁ (Refined − Intermediate)"); _off(axes[0, 5]); _cbar(axes[0, 5], im)

    # ======== DÒNG 2: GÓC 2 ========
    axes[1, 0].imshow(I2, cmap='gray')
    axes[1, 0].set_title("Input Hologram 2"); _off(axes[1, 0])

    axes[1, 1].imshow(I_pred2, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title("Reconstructed Hologram 2"); _off(axes[1, 1])

    # Phổ Fourier của Hologram 2
    fft2 = np.log(np.abs(np.fft.fftshift(np.fft.fft2(I2))) + 1e-3)
    axes[1, 2].imshow(fft2, cmap='viridis', vmin=0, vmax=12)
    axes[1, 2].set_title("Hologram 2 Fourier Spectrum"); _off(axes[1, 2])

    im = axes[1, 3].imshow(phase2, cmap='jet', vmin=-np.pi, vmax=np.pi)
    axes[1, 3].set_title("Refined Phase 2 (Wrapped)"); _off(axes[1, 3]); _cbar(axes[1, 3], im)

    im = axes[1, 4].imshow(phase_rough2, cmap='jet', vmin=-np.pi, vmax=np.pi)
    axes[1, 4].set_title("Intermediate Phase 2 (Wrapped)"); _off(axes[1, 4]); _cbar(axes[1, 4], im)

    delta2 = np.angle(np.exp(1j * (phase2 - phase_rough2)))
    im = axes[1, 5].imshow(delta2, cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    axes[1, 5].set_title("Phase Refinement Δφ₂ (Refined − Intermediate)"); _off(axes[1, 5]); _cbar(axes[1, 5], im)

    # ======== DÒNG 3: ĐÁNH GIÁ ========
    mid_row = phase1.shape[0] // 2

    # Profile giữa ảnh — so sánh pha tinh chỉnh vs trung gian
    axes[2, 0].plot(phase_rough1[mid_row, :], 'b-',  linewidth=1.2, label='Intermediate φ₁')
    axes[2, 0].plot(phase1[mid_row, :],       'r--', linewidth=1.2, label='Refined φ₁')
    if phase_gt is not None:
        axes[2, 0].plot(phase_gt[mid_row, :], 'k:',  linewidth=1.2, label='Ground Truth')
    axes[2, 0].set_title("Phase Profile — Mid-line Comparison (Góc 1)")
    axes[2, 0].legend(fontsize=8); axes[2, 0].grid(True)

    # Profile Góc 2
    axes[2, 1].plot(phase_rough2[mid_row, :], 'c-',  linewidth=1.2, label='Intermediate φ₂')
    axes[2, 1].plot(phase2[mid_row, :],       'm--', linewidth=1.2, label='Refined φ₂')
    axes[2, 1].set_title("Phase Profile — Mid-line Comparison (Góc 2)")
    axes[2, 1].legend(fontsize=8); axes[2, 1].grid(True)

    # So sánh pha tinh chỉnh 1 vs pha tinh chỉnh 2 (mục tiêu: phải giống nhau)
    axes[2, 2].plot(phase1[mid_row, :],  'r-',  linewidth=1.2, label='Refined φ₁')
    axes[2, 2].plot(phase2[mid_row, :],  'b--', linewidth=1.2, label='Refined φ₂')
    axes[2, 2].set_title("Consistency Check: φ₁ vs φ₂ (mục tiêu: trùng nhau)")
    axes[2, 2].legend(fontsize=8); axes[2, 2].grid(True)

    if phase_gt is not None:
        # Error map so với Ground Truth
        err = np.abs(np.angle(np.exp(1j * (phase_gt - phase1))))
        im = axes[2, 3].imshow(err, cmap='hot', vmin=0, vmax=np.pi)
        axes[2, 3].set_title("Error Map vs Ground Truth"); _off(axes[2, 3]); _cbar(axes[2, 3], im)

        mse  = np.mean(err ** 2)
        psnr = 20 * np.log10(2 * np.pi / (np.sqrt(mse) + 1e-8))
        axes[2, 4].text(0.5, 0.5,
            f"MSE  = {mse:.5f}\nPSNR = {psnr:.2f} dB",
            ha='center', va='center', fontsize=14,
            transform=axes[2, 4].transAxes)
        axes[2, 4].set_title("Quantitative Metrics"); _off(axes[2, 4])
    else:
        _off(axes[2, 3]); _off(axes[2, 4])

    # Consistency map: |U1_norm - U2_norm| theo pixel
    U1_norm = np.exp(1j * phase1)
    U2_norm = np.exp(1j * phase2)
    cons_map = np.abs(U1_norm - U2_norm)  # range [0, 2]
    im = axes[2, 5].imshow(cons_map, cmap='hot', vmin=0, vmax=2)
    axes[2, 5].set_title("Consistency Map |U₁ − U₂|\n(mục tiêu: gần 0 toàn ảnh)"); _off(axes[2, 5]); _cbar(axes[2, 5], im)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"visual_evaluation_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   🖼️  Đã lưu kết quả mẫu {sample_idx}: {save_path}")

if __name__ == "__main__":
    evaluate()

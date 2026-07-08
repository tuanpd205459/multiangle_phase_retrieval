import numpy as np
import torch

def compute_physics_loss(U_pred, I_real, k, eps=1e-8):
    """
    Tính Differentiable Physics Loss (L_phys).
    Mô phỏng quá trình giao thoa vật lý cường độ hologram: I_pred = |U_pred + R|^2
    và so sánh L1 với hologram thực tế I_real.
    
    U_pred: Trường sóng phức dự đoán [B, 1, H, W]
    I_real: Hologram cường độ thực tế [B, 1, H, W] (đã chuẩn hóa về [0, 1])
    k: Tần số sóng mang [B, 2] (kx, ky)
    """
    B, C, H, W = U_pred.shape
    device = U_pred.device
    
    # 1. Tạo sóng tham chiếu R tương ứng với kx, ky
    y_grid = torch.arange(H, dtype=torch.float32, device=device)
    x_grid = torch.arange(W, dtype=torch.float32, device=device)
    mesh_y, mesh_x = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    mesh_x_expanded = mesh_x.view(1, 1, H, W)
    mesh_y_expanded = mesh_y.view(1, 1, H, W)
    if k.ndim == 1:
        kx = k[0].expand(B).view(B, 1, 1, 1)
        ky = k[1].expand(B).view(B, 1, 1, 1)
    else:
        kx = k[:, 0].view(B, 1, 1, 1)
        ky = k[:, 1].view(B, 1, 1, 1)
    
    # R = exp(i * phase_carrier)
    phase_carrier = 2.0 * np.pi * (kx * mesh_x_expanded / W + ky * mesh_y_expanded / H)
    R = torch.complex(torch.cos(phase_carrier), torch.sin(phase_carrier))
    
    # 2. Tính hologram cường độ dự đoán
    I_pred = torch.abs(U_pred * R + 1.0)**2
    
    # 3. Khớp tỷ lệ trung bình (Global Scale Matching) thay vì Min-Max Normalization.
    # Tránh hiện tượng bất biến quy mô (scale-invariance) khiến biên độ U bị triệt tiêu về 0 do weight decay.
    # Tính toán hệ số tỷ lệ riêng cho từng mẫu trong Batch [B, 1, 1, 1].
    scale = torch.mean(I_real, dim=(-2, -1), keepdim=True) / (torch.mean(I_pred, dim=(-2, -1), keepdim=True) + eps)
    I_pred_scaled = I_pred * scale
    
    # 4. Tính toán sai lệch L1
    return torch.mean(torch.abs(I_pred_scaled - I_real))

def compute_complex_consistency_loss(U1, U2, eps=1e-8):
    """
    Tính Complex Consistency Loss (L_cons).
    Đồng bộ pha toàn cục (global phase offset delta_psi) giữa hai nhánh đa góc 
    và ép nhất quán pha trực tiếp trong miền phức sử dụng unit phasor để tránh wrapping.
    
    U1, U2: Trường sóng phức từ 2 nhánh Siamese [B, 1, H, W]
    """
    B, C, H, W = U1.shape
    
    # 1. Tính toán lệch pha toàn cục delta_psi giữa U1 và U2 bằng tích chéo
    # delta_psi = angle( sum( U1 * conj(U2) ) )
    cross_prod = U1 * torch.conj(U2)
    sum_cross = torch.sum(cross_prod, dim=(-2, -1), keepdim=True) # [B, 1, 1, 1]
    delta_psi = torch.angle(sum_cross)
    
    # 2. Bù lệch pha toàn cục cho U2
    cos_psi = torch.cos(delta_psi)
    sin_psi = torch.sin(delta_psi)
    exp_psi = torch.complex(cos_psi, sin_psi)
    U2_aligned = U2 * exp_psi
    
    # 3. Chuẩn hóa biên độ để so sánh nhất quán pha thuần khiết (unit phasor)
    # Unit phasor: exp(i*phi) = U / |U|
    U1_norm = U1 / (torch.abs(U1) + eps)
    U2_norm = U2_aligned / (torch.abs(U2_aligned) + eps)
    
    # 4. Tính sai lệch L1 trong miền phức
    return torch.mean(torch.abs(U1_norm - U2_norm))

def compute_total_variation_loss(phase):
    """
    Tính Total Variation (TV) Loss trên pha để loại bỏ nhiễu speckle 
    và làm phẳng các vùng pha nền.
    
    phase: Tensor pha khôi phục [B, 1, H, W]
    """
    # Tính sai phân hữu hạn theo hai chiều x và y
    diff_y = torch.abs(phase[:, :, 1:, :] - phase[:, :, :-1, :])
    diff_x = torch.abs(phase[:, :, :, 1:] - phase[:, :, :, :-1])
    
    return torch.mean(diff_y) + torch.mean(diff_x)

def compute_total_loss(U1, U2, I1, I2, k1, k2, config):
    """
    Tổng hợp các hàm loss với các trọng số cấu hình.
    """
    # 1. Physics Loss cho từng nhánh
    loss_phys1 = compute_physics_loss(U1, I1, k1)
    loss_phys2 = compute_physics_loss(U2, I2, k2)
    loss_phys = loss_phys1 + loss_phys2
    
    # 2. Consistency Loss giữa hai góc
    loss_cons = compute_complex_consistency_loss(U1, U2)
    
    # 3. Background Sparsity / Smoothness (TV Loss) trên pha và biên độ khôi phục
    phase1 = torch.angle(U1)
    phase2 = torch.angle(U2)
    amp1 = torch.abs(U1)
    amp2 = torch.abs(U2)
    
    loss_tv_phase = compute_total_variation_loss(phase1) + compute_total_variation_loss(phase2)
    loss_tv_amp = compute_total_variation_loss(amp1) + compute_total_variation_loss(amp2)
    
    # Lấy các trọng số từ cấu hình config
    lambda_phys = config['loss'].get('lambda_physics', 1.0)
    lambda_cons = config['loss'].get('lambda_consistency', 0.5)
    lambda_sparsity = config['loss'].get('lambda_sparsity', 0.01)
    lambda_amp = config['loss'].get('lambda_amp_smooth', 0.1)
    
    total_loss = (lambda_phys * loss_phys + 
                  lambda_cons * loss_cons + 
                  lambda_sparsity * loss_tv_phase + 
                  lambda_amp * loss_tv_amp)
                  
    return total_loss, {
        'loss_phys': loss_phys.item(),
        'loss_cons': loss_cons.item(),
        'loss_tv': loss_tv_phase.item(),
        'loss_tv_amp': loss_tv_amp.item(),
        'total_loss': total_loss.item()
    }

if __name__ == "__main__":
    print("⏳ Đang kiểm tra các hàm Loss vật lý...")
    
    # Giả lập dữ liệu
    U1 = torch.complex(torch.rand(2, 1, 256, 256), torch.rand(2, 1, 256, 256))
    U2 = torch.complex(torch.rand(2, 1, 256, 256), torch.rand(2, 1, 256, 256))
    
    I1 = torch.rand(2, 1, 256, 256)
    I2 = torch.rand(2, 1, 256, 256)
    
    k1 = torch.tensor([[40.0, -30.0], [40.0, -30.0]])
    k2 = torch.tensor([[-45.0, -35.0], [-45.0, -35.0]])
    
    config = {
        'loss': {
            'lambda_physics': 1.0,
            'lambda_consistency': 0.5,
            'lambda_sparsity': 0.01
        }
    }
    
    total_loss, loss_dict = compute_total_loss(U1, U2, I1, I2, k1, k2, config)
    
    print("✅ Kiểm tra thành công!")
    print(f"Total Loss value: {total_loss.item():.4f}")
    print(f"Physics Loss component: {loss_dict['loss_phys']:.4f}")
    print(f"Consistency Loss component: {loss_dict['loss_cons']:.4f}")
    print(f"TV Loss component: {loss_dict['loss_tv']:.4f}")

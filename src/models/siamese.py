import os
import sys
import torch
import torch.nn as nn

# Thêm thư mục gốc của dự án vào sys.path để chạy trực tiếp không bị lỗi import src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.demodulator import DifferentiableDemodulator
from src.models.unet import PhaseRefiningUNet


# ==============================================================================
# KIỪN TRÚC SIAMESE CHÍNH
# (KEstimator CNN đã được thay thế bằng Spectral Centroid trong Demodulator)
# ==============================================================================

class SiameseTeacherModel(nn.Module):
    """
    Kiến trúc Siamese gồm 2 nhánh giải điều chế khả vi (Shared weights U-Net).

    Luồng xử lý (Spectral Centroid thay thế KEstimator):
        1. Demodulator nhận Hologram + k_dataset
        2. Bước 1: Dịch phổ sơ bộ → FFT → tính Spectral Centroid → Δk
        3. Bước 2: k_final = k_dataset + Δk → dịch phổ lại → IFFT → U_rough
        4. U-Net tinh chỉnh trường sóng thô → trường sóng sạch
        5. Consistency Loss ép 2 nhánh hội tụ về cùng phi_object

    Ưu điểm kiến trúc mới:
        - Không có Gauge Ambiguity: centroid tự neo vào tọa độ Fourier tuyệt đối
        - Nhẹ hơn: loại bỏ toàn bộ CNN KEstimator (4 Conv + GAP + MLP)
        - Chính xác hơn: Δk tính trực tiếp từ vật lý, không phải học từ dữ liệu
    """
    def __init__(self,
                 filter_radius: float = 50.0):
        super(SiameseTeacherModel, self).__init__()

        # Khởi tạo bán kính bộ lọc ellipse
        filter_radius_x = filter_radius * 0.4
        filter_radius_y = filter_radius * 1.2

        # Module giải điều chế khả vi (tích hợp Spectral Centroid Correction)
        self.demodulator = DifferentiableDemodulator(
            filter_radius_x=filter_radius_x,
            filter_radius_y=filter_radius_y
        )

        # Mạng U-Net chung trọng số (Shared Weights) tinh chỉnh pha
        self.unet = PhaseRefiningUNet()

    def forward_single_branch(self, I, k_param, mask_override=None):
        """
        Xử lý một nhánh (một góc chiếu).

        Args:
            I:            Hologram cường độ [B, 1, H, W]
            k_param:      Sóng mang thô từ dataset [B, 2] hoặc [2]
            mask_override: Gaussian mask thích nghi [B, 1, H, W] hoặc None

        Returns:
            U_refined:    Trường sóng phức tinh chỉnh [B, 1, H, W]
            amplitude:    Biên độ [B, 1, H, W]
            phase:        Pha sạch [B, 1, H, W]
            phase_rough:  Pha sơ bộ (trước U-Net) [B, 1, H, W]
            k_final:      Tần số sóng mang sau hiệu chỉnh Spectral Centroid [B, 2]
            delta_k:      Độ lệch Δk tính từ Spectral Centroid [B, 2]
        """
        B = I.shape[0]

        # Chuẩn hóa k_param về [B, 2]
        if k_param.ndim == 1:
            k_param = k_param.unsqueeze(0).expand(B, -1)

        kx = k_param[:, 0]
        ky = k_param[:, 1]

        # Demodulator: dịch phổ sơ bộ → Spectral Centroid → hiệu chỉnh k → IFFT
        U_rough, delta_k, k_final = self.demodulator(I, kx, ky, mask_override=mask_override)

        # Pha sơ bộ (kết quả của Demodulator đã hiệu chỉnh)
        phase_rough = torch.angle(U_rough)  # [B, 1, H, W]

        # Tinh chỉnh bằng U-Net
        U_refined = self.unet(U_rough)      # [B, 1, H, W] phức

        # Trích xuất biên độ và pha sạch
        amplitude = torch.abs(U_refined)
        phase     = torch.angle(U_refined)

        return U_refined, amplitude, phase, phase_rough, k_final, delta_k

    def forward(self, I1, k1, I2, k2, mask1=None, mask2=None):
        """
        Xử lý song song hai nhánh Siamese cho hai góc chiếu khác nhau.

        Returns:
            Tuple cho nhánh 1: (U1, amp1, phase1, phase_rough1, k1_final, delta_k1)
            Tuple cho nhánh 2: (U2, amp2, phase2, phase_rough2, k2_final, delta_k2)
        """
        branch1 = self.forward_single_branch(I1, k1, mask_override=mask1)
        branch2 = self.forward_single_branch(I2, k2, mask_override=mask2)
        return branch1, branch2

    def forward(self, I1, k1, I2, k2, mask1=None, mask2=None):
        """
        Xử lý song song hai nhánh Siamese cho hai góc chiếu khác nhau.

        Returns:
            Tuple cho nhánh 1: (U1, amp1, phase1, phase_rough1, k1_final, delta_k1)
            Tuple cho nhánh 2: (U2, amp2, phase2, phase_rough2, k2_final, delta_k2)
        """
        branch1 = self.forward_single_branch(I1, k1, mask_override=mask1)
        branch2 = self.forward_single_branch(I2, k2, mask_override=mask2)
        return branch1, branch2


# ==============================================================================
# BẢN KIỂM TRA CHỨC NĂNG (TEST RUN)
# ==============================================================================

if __name__ == "__main__":
    print("⏳ Đang kiểm tra mô hình Siamese với Spectral Centroid Correction...")

    model = SiameseTeacherModel(filter_radius=50.0)

    # Tạo tensor giả lập cho 2 góc chiếu
    B, H, W = 2, 256, 256
    I1 = torch.rand(B, 1, H, W)
    I2 = torch.rand(B, 1, H, W)

    # Sóng mang thô từ dataset (per-sample)
    k1_dummy = torch.tensor([[40.0, -30.0], [42.0, -28.0]], requires_grad=True)
    k2_dummy = torch.tensor([[-45.0, -35.0], [-43.0, -37.0]], requires_grad=True)

    (U1, amp1, phase1, phase_rough1, k1_final, dk1), \
    (U2, amp2, phase2, phase_rough2, k2_final, dk2) = model(I1, k1_dummy, I2, k2_dummy)

    # Tính Loss giả lập và lan truyền ngược để kiểm tra gradient
    loss = torch.mean(torch.abs(U1) + torch.abs(U2))
    loss.backward()

    print("✅ Kiểm tra chạy thử thành công!\n")
    print(f"   U1 shape: {U1.shape}, dtype: {U1.dtype}")
    print(f"   U2 shape: {U2.shape}, dtype: {U2.dtype}")
    print(f"\n📊 Δk từ Spectral Centroid (không cần train):")
    print(f"   delta_k1 = {dk1.detach().numpy()}")
    print(f"   delta_k2 = {dk2.detach().numpy()}")
    print(f"\n📍 k_final (k_dataset + Δk_centroid):")
    print(f"   k1_final = {k1_final.detach().numpy()}")
    print(f"   k2_final = {k2_final.detach().numpy()}")
    print(f"\n🔬 Kiểm tra Gradient:")
    print(f"   filter_radius_x grad: {model.demodulator.filter_radius_x.grad.item():.6f}")
    print(f"   filter_radius_y grad: {model.demodulator.filter_radius_y.grad.item():.6f}")
    print("\n✅ Spectral Centroid hoạt động đúng — không cần CNN KEstimator!")

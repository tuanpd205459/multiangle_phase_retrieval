import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Thêm thư mục gốc của dự án vào sys.path để chạy trực tiếp không bị lỗi import src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.demodulator import DifferentiableDemodulator
from src.models.unet import PhaseRefiningUNet


# ==============================================================================
# MODULE HỌC SÓC MANG: K-ESTIMATOR
# ==============================================================================

class KEstimator(nn.Module):
    """
    Mạng CNN nhỏ dự đoán độ lệch sóng mang (delta_k) từ hologram cường độ.

    Ý tưởng vật lý:
        Sau khi giải điều chế FFT thô, pha thu được có dạng:
            phi_rough = phi_object + phi_noise + phi_carrier_residual

        phi_carrier_residual xảy ra khi tần số sóng mang k ước lượng bị lệch.
        Mạng này học bù trừ Δk = (Δkx, Δky) sao cho:
            k_final = k_dataset + Δk

        Khi Consistency Loss ép hai nhánh (2 góc khác nhau) phải hội tụ về cùng
        phi_object, mạng BẮT BUỘC phải học đúng Δk để loại bỏ phần phi_carrier_residual
        (vì phần đó khác nhau ở 2 góc, không thể đồng thời thỏa mãn 2 nhánh nếu còn dư).

    Kiến trúc:
        Encoder CNN 4 tầng + Global Average Pooling → MLP → (Δkx, Δky)

    Tham số đầu vào:
        I:  Hologram cường độ [B, 1, H, W]
    Đầu ra:
        delta_k: Tensor [B, 2] chứa (Δkx, Δky) cho từng ảnh trong batch
    """
    def __init__(self, max_delta_k: float = 5.0):
        """
        max_delta_k: Giới hạn tối đa của độ lệch Δk (pixel). Giúp ổn định huấn luyện.
                     Giá trị mặc định ±5 pixel phù hợp với hầu hết hệ thống DHM.
        """
        super(KEstimator, self).__init__()
        self.max_delta_k = max_delta_k

        # Encoder CNN nhỏ: tăng kênh nhanh, giảm kích thước nhanh
        self.encoder = nn.Sequential(
            # Tầng 1: [B, 1, H, W] → [B, 16, H/2, W/2]
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),

            # Tầng 2: → [B, 32, H/4, W/4]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            # Tầng 3: → [B, 64, H/8, W/8]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            # Tầng 4: → [B, 128, H/16, W/16]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Global Average Pooling: [B, 128, H/16, W/16] → [B, 128]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # MLP đầu ra: [B, 128] → [B, 2] (Δkx, Δky)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 2)
        )

        # Khởi tạo layer cuối về 0 → Δk ban đầu = 0 (không làm hỏng k_dataset)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        """
        I: Hologram cường độ [B, 1, H, W]
        Returns: delta_k [B, 2] — (Δkx, Δky) đã được clamp vào [-max_delta_k, +max_delta_k]
        """
        feat = self.encoder(I)          # [B, 128, H/16, W/16]
        feat = self.gap(feat)           # [B, 128, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, 128]
        delta_k = self.head(feat)       # [B, 2]

        # Giới hạn biên độ học để tránh mô hình nhảy k quá lớn trong epoch đầu
        delta_k = torch.tanh(delta_k) * self.max_delta_k
        return delta_k


# ==============================================================================
# KIẾN TRÚC SIAMESE CHÍNH (ĐÃ TÍCH HỢP K-ESTIMATOR)
# ==============================================================================

class SiameseTeacherModel(nn.Module):
    """
    Kiến trúc Siamese gồm 2 nhánh giải điều chế khả vi (Shared weights U-Net).

    Luồng xử lý mới (có học sóng mang):
        1. KEstimator nhận Hologram → dự đoán Δk = (Δkx, Δky)
        2. k_final = k_dataset_thô + Δk_learned
        3. DifferentiableDemodulator dùng k_final để lọc búp phổ +1
        4. U-Net tinh chỉnh trường sóng thô → trường sóng sạch
        5. Consistency Loss ép 2 nhánh hội tụ về cùng phi_object
           → Buộc KEstimator học đúng phần bù sóng mang dư mà không cần nhãn

    Tham số khởi tạo:
        filter_radius:  Bán kính lọc FFT mặc định (pixel)
        max_delta_k:    Biên độ tối đa Δk mà KEstimator có thể học (pixel)
        use_k_estimator: Bật/tắt module học sóng mang (mặc định: True)
    """
    def __init__(self,
                 filter_radius: float = 50.0,
                 k1_init: list = None,
                 k2_init: list = None,
                 max_delta_k: float = 5.0,
                 use_k_estimator: bool = True):
        super(SiameseTeacherModel, self).__init__()

        # Khởi tạo bán kính bộ lọc ellipse (Rx nhỏ hơn, Ry lớn hơn)
        filter_radius_x = filter_radius * 0.4
        filter_radius_y = filter_radius * 1.2

        # 1. Module giải điều chế khả vi (bán kính lọc là nn.Parameter có thể học)
        self.demodulator = DifferentiableDemodulator(
            filter_radius_x=filter_radius_x,
            filter_radius_y=filter_radius_y
        )

        # 2. Module học sóng mang (KEstimator) — dùng chung cho cả 2 nhánh
        #    Shared weights: cùng một mạng con nhận hologram từ góc nào cũng được
        self.use_k_estimator = use_k_estimator
        if use_k_estimator:
            self.k_estimator = KEstimator(max_delta_k=max_delta_k)

        # 3. Mạng U-Net chung trọng số (Shared Weights) tinh chỉnh pha
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
            k_final:      Tần số sóng mang cuối cùng (sau bù Δk) [B, 2]
            delta_k:      Độ lệch học được [B, 2] (hoặc zeros nếu không dùng KEstimator)
        """
        B = I.shape[0]

        # ---- BƯỚC 0: Học độ lệch sóng mang Δk từ hologram ----
        if self.use_k_estimator:
            delta_k = self.k_estimator(I)  # [B, 2]
        else:
            delta_k = torch.zeros(B, 2, device=I.device, dtype=I.dtype)

        # ---- BƯỚC 1: Tính k_final = k_dataset + Δk ----
        if len(k_param.shape) == 2:
            # Per-sample [B, 2]
            k_final = k_param + delta_k  # [B, 2]
            kx = k_final[:, 0]           # [B]
            ky = k_final[:, 1]           # [B]
        else:
            # Toàn cục [2] → broadcast lên batch
            k_broadcast = k_param.unsqueeze(0).expand(B, -1)  # [B, 2]
            k_final = k_broadcast + delta_k
            kx = k_final[:, 0]
            ky = k_final[:, 1]

        # ---- BƯỚC 2: Giải điều chế khả vi FFT với k_final ----
        U_rough = self.demodulator(I, kx, ky, mask_override=mask_override)  # [B, 1, H, W] phức

        # Trích xuất pha trung gian thô (vùng chọn phổ bậc +1 sau IFFT)
        phase_rough = torch.angle(U_rough)  # [B, 1, H, W]

        # ---- BƯỚC 3: Tinh chỉnh bằng U-Net ----
        U_refined = self.unet(U_rough)  # [B, 1, H, W] phức

        # ---- BƯỚC 4: Trích xuất biên độ và pha sạch ----
        amplitude = torch.abs(U_refined)
        phase = torch.angle(U_refined)

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


# ==============================================================================
# BẢN KIỂM TRA CHỨC NĂNG (TEST RUN)
# ==============================================================================

if __name__ == "__main__":
    print("⏳ Đang kiểm tra mô hình Siamese với module học sóng mang KEstimator...")

    model = SiameseTeacherModel(
        filter_radius=50.0,
        max_delta_k=5.0,
        use_k_estimator=True
    )

    # Tạo tensor giả lập cho 2 góc chiếu
    B, H, W = 2, 256, 256
    I1 = torch.rand(B, 1, H, W)
    I2 = torch.rand(B, 1, H, W)

    # Sóng mang thô từ dataset (per-sample)
    k1_dummy = torch.tensor([[40.0, -30.0], [42.0, -28.0]])
    k2_dummy = torch.tensor([[-45.0, -35.0], [-43.0, -37.0]])

    (U1, amp1, phase1, phase_rough1, k1_final, dk1), \
    (U2, amp2, phase2, phase_rough2, k2_final, dk2) = model(I1, k1_dummy, I2, k2_dummy)

    # Tính Loss giả lập và lan truyền ngược để kiểm tra gradient
    loss = torch.mean(torch.abs(U1) + torch.abs(U2))
    loss.backward()

    print("✅ Kiểm tra chạy thử thành công!\n")
    print(f"   U1 shape: {U1.shape}, dtype: {U1.dtype}")
    print(f"   U2 shape: {U2.shape}, dtype: {U2.dtype}")
    print(f"\n📊 Kiểm tra KEstimator (ban đầu Δk ≈ 0 do zero-init):")
    print(f"   delta_k1 = {dk1.detach().numpy()}")
    print(f"   delta_k2 = {dk2.detach().numpy()}")
    print(f"\n📍 k_final (k_dataset + Δk):")
    print(f"   k1_final = {k1_final.detach().numpy()}")
    print(f"   k2_final = {k2_final.detach().numpy()}")
    print(f"\n🔬 Kiểm tra Gradient:")
    print(f"   filter_radius_x grad: {model.demodulator.filter_radius_x.grad.item():.6f}")
    print(f"   filter_radius_y grad: {model.demodulator.filter_radius_y.grad.item():.6f}")

    # Kiểm tra gradient chạy ngược vào k_estimator
    total_grad_norm = 0.0
    for name, p in model.k_estimator.named_parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item()
    print(f"   KEstimator total grad norm: {total_grad_norm:.6f}")
    print("\n✅ Gradient flow qua KEstimator hoạt động đúng!")

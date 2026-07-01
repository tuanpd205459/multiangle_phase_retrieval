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

class SiameseTeacherModel(nn.Module):
    """
    Kiến trúc Siamese gồm 2 nhánh giải điều chế khả vi (Shared weights U-Net).
    
    Mô hình nhận vào 2 hologram cường độ (I1, I2) và 2 tần số sóng mang tương ứng (k1, k2).
    - Giải điều chế khả vi độc lập để đưa búp sóng +1 về DC thu được 2 trường sóng thô.
    - Đưa 2 trường sóng thô qua mạng U-Net chung trọng số để tinh chỉnh biên độ và pha.
    - Xuất ra 2 trường sóng phức tinh chỉnh cùng với biên độ và pha tương ứng.
    """
    def __init__(self, filter_radius=50):
        super(SiameseTeacherModel, self).__init__()
        
        # 1. Module giải điều chế khả vi (Không chứa tham số cần học của mạng nơ-ron)
        self.demodulator = DifferentiableDemodulator(filter_radius=filter_radius)
        
        # 2. Mạng U-Net chung trọng số (Shared Weights)
        self.unet = PhaseRefiningUNet()

    def forward_single_branch(self, I, k):
        """
        Xử lý đơn nhánh (đối với một góc chiếu)
        I: Hologram [B, 1, H, W]
        k: Tần số sóng mang [B, 2] (kx, ky)
        """
        # Tách kx và ky
        kx = k[:, 0]
        ky = k[:, 1]
        
        # 1. Giải điều chế khả vi miền Fourier
        U_rough = self.demodulator(I, kx, ky) # [B, 1, H, W] (phức)
        
        # 2. Tinh chỉnh bằng mạng U-Net
        U_refined = self.unet(U_rough)        # [B, 1, H, W] (phức)
        
        # 3. Trích xuất biên độ và pha của trường sóng khôi phục
        amplitude = torch.abs(U_refined)
        phase = torch.angle(U_refined)
        
        return U_refined, amplitude, phase

    def forward(self, I1, k1, I2, k2):
        """
        Xử lý song song hai nhánh Siamese cho hai góc chiếu khác nhau.
        """
        # Nhánh 1: Góc chiếu thứ nhất
        U1, amp1, phase1 = self.forward_single_branch(I1, k1)
        
        # Nhánh 2: Góc chiếu thứ hai
        U2, amp2, phase2 = self.forward_single_branch(I2, k2)
        
        return (U1, amp1, phase1), (U2, amp2, phase2)

if __name__ == "__main__":
    print("⏳ Đang kiểm tra mô hình Siamese...")
    model = SiameseTeacherModel(filter_radius=50)
    
    # Tạo tensor giả lập cho 2 góc chiếu
    I1 = torch.rand(2, 1, 256, 256)
    I2 = torch.rand(2, 1, 256, 256)
    
    k1 = torch.tensor([[40.0, -30.0], [40.0, -30.0]], requires_grad=True)
    k2 = torch.tensor([[-45.0, -35.0], [-45.0, -35.0]], requires_grad=True)
    
    (U1, amp1, phase1), (U2, amp2, phase2) = model(I1, k1, I2, k2)
    
    print("✅ Kiểm tra thành công!")
    print(f"U1 shape: {U1.shape}, dtype: {U1.dtype}")
    print(f"U2 shape: {U2.shape}, dtype: {U2.dtype}")
    print(f"Pha khôi phục 1 shape: {phase1.shape}")
    print(f"Biên độ khôi phục 2 shape: {amp2.shape}")

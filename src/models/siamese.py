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
    def __init__(self, filter_radius=50.0, k1_init=[40.0, -30.0], k2_init=[-45.0, -35.0]):
        super(SiameseTeacherModel, self).__init__()
        
        # Khởi tạo bán kính elip ngang (Rx) và dọc (Ry) dựa trên filter_radius để tránh phá vỡ config
        # Rx = 0.4 * filter_radius (khoảng 20px), Ry = 1.2 * filter_radius (khoảng 60px)
        filter_radius_x = filter_radius * 0.4
        filter_radius_y = filter_radius * 1.2
        
        # 1. Module giải điều chế khả vi (chứa nn.Parameter filter_radius_x và filter_radius_y bên trong)
        self.demodulator = DifferentiableDemodulator(filter_radius_x=filter_radius_x, filter_radius_y=filter_radius_y)
        
        # 2. Tần số sóng mang là nn.Parameter để có thể tối ưu hóa/học được
        self.k1 = nn.Parameter(torch.tensor(k1_init, dtype=torch.float32))
        self.k2 = nn.Parameter(torch.tensor(k2_init, dtype=torch.float32))
        
        # 3. Mạng U-Net chung trọng số (Shared Weights)
        self.unet = PhaseRefiningUNet()

    def forward_single_branch(self, I, k_param, mask_override=None):
        """
        Xử lý đơn nhánh (đối với một góc chiếu) sử dụng tham số sóng mang truyền vào.
        I: Hologram [B, 1, H, W]
        k_param: Sóng mang [B, 2] hoặc nn.Parameter [2] (kx, ky)
        mask_override: Tensor [B, 1, H, W] hoặc None chứa bộ lọc thích nghi
        """
        B = I.shape[0]
        # Hỗ trợ cả sóng mang per-sample [B, 2] và sóng mang toàn cục [2]
        if len(k_param.shape) == 2:
            kx = k_param[:, 0]
            ky = k_param[:, 1]
        else:
            kx = k_param[0].expand(B)
            ky = k_param[1].expand(B)
        
        # 1. Giải điều chế khả vi miền Fourier
        U_rough = self.demodulator(I, kx, ky, mask_override=mask_override) # [B, 1, H, W] (phức)
        
        # 2. Tinh chỉnh bằng mạng U-Net
        U_refined = self.unet(U_rough)        # [B, 1, H, W] (phức)
        
        # 3. Trích xuất biên độ và pha của trường sóng khôi phục
        amplitude = torch.abs(U_refined)
        phase = torch.angle(U_refined)
        
        return U_refined, amplitude, phase

    def forward(self, I1, k1, I2, k2, mask1=None, mask2=None):
        """
        Xử lý song song hai nhánh Siamese cho hai góc chiếu khác nhau.
        Sử dụng tần số sóng mang k1 và k2 đặc trưng cho từng mẫu (batch-specific)
        truyền từ ngoài vào để đảm bảo dịch phổ chính xác cho dữ liệu thực nghiệm.
        """
        # Nhánh 1: Góc chiếu thứ nhất sử dụng k1 truyền vào và mask1
        U1, amp1, phase1 = self.forward_single_branch(I1, k1, mask_override=mask1)
        
        # Nhánh 2: Góc chiếu thứ hai sử dụng k2 truyền vào và mask2
        U2, amp2, phase2 = self.forward_single_branch(I2, k2, mask_override=mask2)
        
        return (U1, amp1, phase1), (U2, amp2, phase2)

if __name__ == "__main__":
    print("⏳ Đang kiểm tra mô hình Siamese với các tham số vật lý học được...")
    # Khởi tạo model với các giá trị mặc định
    model = SiameseTeacherModel(filter_radius=50.0, k1_init=[40.2, -30.1], k2_init=[-44.8, -35.2])
    
    # Tạo tensor giả lập cho 2 góc chiếu
    I1 = torch.rand(2, 1, 256, 256)
    I2 = torch.rand(2, 1, 256, 256)
    
    k1_dummy = torch.zeros(2, 2) # Chỉ dùng để giữ cấu trúc tương thích
    k2_dummy = torch.zeros(2, 2)
    
    (U1, amp1, phase1), (U2, amp2, phase2) = model(I1, k1_dummy, I2, k2_dummy)
    
    # Tính Loss giả lập và lan truyền ngược để kiểm tra gradient
    loss = torch.mean(torch.abs(U1) + torch.abs(U2))
    loss.backward()
    
    print("✅ Kiểm tra chạy thử thành công!")
    print(f"U1 shape: {U1.shape}, dtype: {U1.dtype}")
    print(f"U2 shape: {U2.shape}, dtype: {U2.dtype}")
    print("\n📊 Kiểm tra tính toán Gradient trên các tham số vật lý học được:")
    print(f"   - k1 value: {model.k1.detach().numpy()}, grad: {model.k1.grad.numpy()}")
    print(f"   - k2 value: {model.k2.detach().numpy()}, grad: {model.k2.grad.numpy()}")
    print(f"   - filter_radius_x value: {model.demodulator.filter_radius_x.item():.2f}, grad: {model.demodulator.filter_radius_x.grad.item():.6f}")
    print(f"   - filter_radius_y value: {model.demodulator.filter_radius_y.item():.2f}, grad: {model.demodulator.filter_radius_y.grad.item():.6f}")


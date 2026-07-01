import os
import glob
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def estimate_carrier_frequency(I, search_radius_min=15, search_radius_max=80):
    """
    Ước lượng tự động tần số sóng mang (kx, ky) của hologram bằng cách tìm đỉnh
    của búp sóng bậc +1 trong miền Fourier 2D (tránh vùng DC ở tâm).
    
    I: numpy array [H, W] - hologram cường độ
    search_radius_min: bán kính tối thiểu để tránh búp sóng DC ở tâm
    search_radius_max: bán kính tối đa tìm kiếm sóng mang
    """
    H, W = I.shape
    # 1. Biến đổi Fourier 2D và dịch tâm
    I_fft = np.fft.fftshift(np.fft.fft2(I))
    I_fft_amp = np.abs(I_fft)
    
    # 2. Tạo mặt nạ để lọc bỏ vùng DC ở tâm và các vùng biên quá xa
    y_grid, x_grid = np.ogrid[-H//2:H//2, -W//2:W//2]
    distance = np.sqrt(x_grid**2 + y_grid**2)
    
    # Chỉ tìm kiếm trong dải tần số thích hợp của búp sóng +1
    mask = (distance >= search_radius_min) & (distance <= search_radius_max)
    
    # 3. Tìm vị trí đỉnh cường độ trong vùng tìm kiếm
    masked_amp = I_fft_amp * mask
    
    # Do đối xứng, búp sóng +1 và -1 sẽ đối xứng qua tâm. 
    # Ta chọn búp sóng nằm ở nửa mặt phẳng trên (y < 0) để đồng nhất hướng
    masked_amp[H//2:, :] = 0 # Chỉ tìm ở nửa trên
    
    max_idx = np.argmax(masked_amp)
    peak_y, peak_x = np.unravel_index(max_idx, masked_amp.shape)
    
    # Chuyển đổi tọa độ về dạng pixel dịch chuyển so với tâm DC (0,0)
    ky = float(peak_y - H//2)
    kx = float(peak_x - W//2)
    
    return kx, ky

def generate_synthetic_phase_cell(H, W, rng):
    """
    Sinh pha giả lập mô phỏng các tế bào sinh học mượt mà.
    Sử dụng tổng các đa thức Zernike bậc thấp cho quang sai nền
    và các hàm Gauss ngẫu nhiên cho cấu trúc tế bào.
    """
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y)
    
    # 1. Quang sai nền (Background aberrations)
    r2 = X**2 + Y**2
    a = rng.uniform(-0.5, 0.5, 5)
    phi_bg = (a[0]*X + a[1]*Y + a[2]*r2 + a[3]*(X**2 - Y**2) + a[4]*2*X*Y)
    phi_bg = phi_bg - np.mean(phi_bg)
    
    # 2. Sinh các cấu trúc tế bào (Gaussian blobs)
    num_cells = rng.integers(1, 4)
    phi_cells = np.zeros_like(X)
    for _ in range(num_cells):
        cx = rng.uniform(-0.4, 0.4)
        cy = rng.uniform(-0.4, 0.4)
        sigma_x = rng.uniform(0.12, 0.25)
        sigma_y = rng.uniform(0.12, 0.25)
        height = rng.uniform(1.5, 5.0) * np.pi  # Pha cao từ 1.5pi đến 5pi
        
        g = height * np.exp(-(((X - cx)**2) / (2 * sigma_x**2) + ((Y - cy)**2) / (2 * sigma_y**2)))
        phi_cells += g
        
    phi = phi_bg + phi_cells
    return phi.astype(np.float32)

def generate_hologram_pair(phi, H, W, rng, noise_level=0.03):
    """
    Sinh cặp hologram cường độ từ cùng một phân phối pha phi ở 2 góc chiếu khác nhau.
    """
    # 1. Tạo lưới tọa độ miền thực
    y = np.arange(H)
    x = np.arange(W)
    X, Y = np.meshgrid(x, y)
    
    # 2. Thiết lập sóng mang ngẫu nhiên cho 2 góc chiếu (đảm bảo búp sóng tách biệt miền Fourier)
    # Góc 1: Thường nằm ở góc phần tư thứ nhất (kx > 0, ky < 0)
    kx1 = rng.uniform(35.0, 50.0)
    ky1 = rng.uniform(-45.0, -30.0)
    
    # Góc 2: Thường nằm ở góc phần tư thứ hai (kx < 0, ky < 0)
    kx2 = rng.uniform(-50.0, -35.0)
    ky2 = rng.uniform(-45.0, -30.0)
    
    # Sóng tham chiếu R = exp(i * phase_carrier)
    phase_carrier1 = 2 * np.pi * (kx1 * X / W + ky1 * Y / H)
    phase_carrier2 = 2 * np.pi * (kx2 * X / W + ky2 * Y / H)
    
    # Vật thể truyền qua thuần pha: U = exp(i * phi)
    U_obj = np.exp(1j * phi)
    
    R1 = np.exp(1j * phase_carrier1)
    R2 = np.exp(1j * phase_carrier2)
    
    # Cường độ giao thoa I = |U + R|^2
    I1 = np.abs(U_obj + R1)**2
    I2 = np.abs(U_obj + R2)**2
    
    # Thêm nhiễu Gauss mô phỏng camera thực tế
    I1 += rng.normal(0, noise_level, size=I1.shape)
    I2 += rng.normal(0, noise_level, size=I2.shape)
    
    # Chuẩn hóa ảnh về dải [0, 1]
    I1 = (I1 - I1.min()) / (I1.max() - I1.min() + 1e-8)
    I2 = (I2 - I2.min()) / (I2.max() - I2.min() + 1e-8)
    
    return I1.astype(np.float32), I2.astype(np.float32), (kx1, ky1), (kx2, ky2)


class MultiAngleHologramDataset(Dataset):
    """
    Bộ dữ liệu PyTorch hỗ trợ đồng thời sinh dữ liệu mô phỏng (Synthetic)
    và đọc ảnh thực nghiệm thực tế (.bmp, .tif, .png) có tự động ước lượng sóng mang.
    """
    def __init__(self, mode='synthetic', data_dir=None, num_samples=3000, 
                 image_size=(256, 256), seed=42, transform=None):
        self.mode = mode.lower()
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.H, self.W = image_size
        self.transform = transform
        
        if self.mode == 'synthetic':
            self.rng = np.random.default_rng(seed)
            # Tạo sẵn các seed con cho từng mẫu để đảm bảo tính tái lập (reproducible)
            self.sample_seeds = self.rng.integers(0, 2**32 - 1, size=num_samples)
        else:
            # Chế độ đọc dữ liệu thực tế
            if not data_dir or not os.path.exists(data_dir):
                raise ValueError(f"Đường dẫn dữ liệu thực tế không hợp lệ: {data_dir}")
            self.pairs = self._find_real_pairs(data_dir)
            if len(self.pairs) == 0:
                print(f"⚠️ Cảnh báo: Không tìm thấy cặp ảnh hologram phù hợp trong {data_dir}!")
            self.num_samples = len(self.pairs)
            
    def _find_real_pairs(self, data_dir):
        """
        Tìm và ghép cặp các ảnh hologram thực tế.
        Quy tắc đặt tên file ảnh ghép cặp được định nghĩa:
        - Mẫu: 'xxxx_angle1.png' và 'xxxx_angle2.png'
        - Hoặc: 'xxxx_1.tif' và 'xxxx_2.tif'
        - Hoặc: 'xxxx_goc1.bmp' và 'xxxx_goc2.bmp'
        """
        extensions = ['*.bmp', '*.tif', '*.tiff', '*.png']
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(data_dir, ext)))
            all_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
            
        all_files = sorted(list(set(all_files)))
        pairs = []
        
        # Mẫu regex nhận diện góc chiếu
        # Group 1: tiền tố chung của mẫu (prefix)
        # Group 2: chỉ số góc (1 hoặc 2, hoặc angle1/angle2, goc1/goc2)
        pattern = re.compile(r'^(.*?)(?:_angle|_goc|_)?([12])\.[a-zA-Z0-9]+$')
        
        prefix_dict = {}
        for fpath in all_files:
            fname = os.path.basename(fpath)
            match = pattern.match(fname)
            if match:
                prefix = match.group(1)
                angle = int(match.group(2))
                if prefix not in prefix_dict:
                    prefix_dict[prefix] = {}
                prefix_dict[prefix][angle] = fpath
                
        # Duyệt qua các prefix tìm được để ghép cặp
        for prefix, angles in prefix_dict.items():
            if 1 in angles and 2 in angles:
                pairs.append((angles[1], angles[2]))
                
        # Nếu không tìm thấy cặp theo regex, tự động ghép các ảnh liên tiếp trong danh sách
        if len(pairs) == 0 and len(all_files) >= 2:
            print("💡 Không tìm thấy hậu tố ghép cặp (_1/_2 hoặc _angle1/_angle2). Ghép cặp tuần tự các file ảnh...")
            for i in range(0, len(all_files) - 1, 2):
                pairs.append((all_files[i], all_files[i+1]))
                
        return pairs
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        if self.mode == 'synthetic':
            # Khởi tạo bộ sinh số ngẫu nhiên cục bộ cho từng mẫu để đảm bảo tính nhất quán
            seed = int(self.sample_seeds[idx])
            local_rng = np.random.default_rng(seed)
            
            # Sinh pha vật thể giả lập
            phi = generate_synthetic_phase_cell(self.H, self.W, local_rng)
            
            # Sinh cặp hologram và sóng mang
            I1, I2, (kx1, ky1), (kx2, ky2) = generate_hologram_pair(phi, self.H, self.W, local_rng)
            
            # Đưa về Tensor PyTorch [1, H, W]
            I1_tensor = torch.from_numpy(I1).unsqueeze(0)
            I2_tensor = torch.from_numpy(I2).unsqueeze(0)
            phi_tensor = torch.from_numpy(phi).unsqueeze(0)
            
            # Chuyển đổi tần số sóng mang thành Tensor
            k1_tensor = torch.tensor([kx1, ky1], dtype=torch.float32)
            k2_tensor = torch.tensor([kx2, ky2], dtype=torch.float32)
            
            return {
                'I1': I1_tensor,
                'I2': I2_tensor,
                'k1': k1_tensor,
                'k2': k2_tensor,
                'phi_gt': phi_tensor # Dữ liệu nhãn chỉ dùng để tính toán chỉ số đánh giá (validation/test)
            }
        else:
            # Chế độ đọc dữ liệu thực tế
            img1_path, img2_path = self.pairs[idx]
            
            # Đọc ảnh cường độ không thay đổi bit depth (hỗ trợ tiff 16-bit)
            I1_raw = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
            I2_raw = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
            
            # Đưa về kích thước chuẩn của hệ thống
            if I1_raw.shape != (self.H, self.W):
                I1_raw = cv2.resize(I1_raw, (self.W, self.H), interpolation=cv2.INTER_AREA)
            if I2_raw.shape != (self.H, self.W):
                I2_raw = cv2.resize(I2_raw, (self.W, self.H), interpolation=cv2.INTER_AREA)
                
            I1 = I1_raw.astype(np.float32)
            I2 = I2_raw.astype(np.float32)
            
            # Chuẩn hóa ảnh dựa trên kiểu dữ liệu thực tế (8-bit hoặc 16-bit)
            max_val1 = 65535.0 if I1_raw.dtype == np.uint16 else 255.0
            max_val2 = 65535.0 if I2_raw.dtype == np.uint16 else 255.0
            
            I1 /= max_val1
            I2 /= max_val2
            
            # Tự động ước lượng tần số sóng mang bằng biến đổi Fourier 2D
            kx1, ky1 = estimate_carrier_frequency(I1)
            kx2, ky2 = estimate_carrier_frequency(I2)
            
            I1_tensor = torch.from_numpy(I1).unsqueeze(0)
            I2_tensor = torch.from_numpy(I2).unsqueeze(0)
            
            k1_tensor = torch.tensor([kx1, ky1], dtype=torch.float32)
            k2_tensor = torch.tensor([kx2, ky2], dtype=torch.float32)
            
            return {
                'I1': I1_tensor,
                'I2': I2_tensor,
                'k1': k1_tensor,
                'k2': k2_tensor,
                'phi_gt': torch.zeros_like(I1_tensor) # Thực tế không có nhãn pha
            }

# ==============================================================================
# CHẠY THỬ NGHIỆM ĐỂ KIỂM TRA CHỨC NĂNG
# ==============================================================================
if __name__ == "__main__":
    print("⏳ Đang kiểm tra chức năng sinh dữ liệu giả lập...")
    dataset = MultiAngleHologramDataset(mode='synthetic', num_samples=5, image_size=(256, 256))
    sample = dataset[0]
    
    print("✅ Kiểm tra thành công!")
    print(f"Hologram 1 shape: {sample['I1'].shape}, Dải giá trị: [{sample['I1'].min():.2f}, {sample['I1'].max():.2f}]")
    print(f"Hologram 2 shape: {sample['I2'].shape}, Dải giá trị: [{sample['I2'].min():.2f}, {sample['I2'].max():.2f}]")
    print(f"Sóng mang góc 1 (kx1, ky1): {sample['k1'].numpy()}")
    print(f"Sóng mang góc 2 (kx2, ky2): {sample['k2'].numpy()}")
    print(f"Pha Ground Truth shape: {sample['phi_gt'].shape}")

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
    """
    H, W = I.shape
    I_fft = np.fft.fftshift(np.fft.fft2(I))
    I_fft_amp = np.abs(I_fft)
    
    y_grid, x_grid = np.ogrid[-H//2:H//2, -W//2:W//2]
    distance = np.sqrt(x_grid**2 + y_grid**2)
    
    mask = (distance >= search_radius_min) & (distance <= search_radius_max)
    masked_amp = I_fft_amp * mask
    masked_amp[H//2:, :] = 0 # Chỉ tìm ở nửa trên
    
    max_idx = np.argmax(masked_amp)
    peak_y, peak_x = np.unravel_index(max_idx, masked_amp.shape)
    
    ky = float(peak_y - H//2)
    kx = float(peak_x - W//2)
    
    return kx, ky

def generate_synthetic_phase_cell(H, W, rng):
    """
    Sinh pha giả lập mô phỏng tế bào sinh học mượt mà (Zernike + Gauss).
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
        height = rng.uniform(1.5, 5.0) * np.pi
        
        g = height * np.exp(-(((X - cx)**2) / (2 * sigma_x**2) + ((Y - cy)**2) / (2 * sigma_y**2)))
        phi_cells += g
        
    phi = phi_bg + phi_cells
    return phi.astype(np.float32)

def generate_hologram_pair_realistic(phi, H, W, rng, noise_level=0.03, objective_na_radius=100):
    """
    MÔ PHỎNG VẬT LÝ QUANG HỌC TIỆM CẬN THỰC NGHIỆM:
    1. Giới hạn độ phân giải quang học bởi khẩu độ vật kính (Pupil function).
    2. Hồ sơ chùm sáng Gaussian (Gaussian Beam Profile) không đều.
    3. Hấp thụ biên độ tế bào không đồng nhất (Amplitude Absorption).
    4. Sóng mang thập phân (Sub-pixel Carrier) gây rò rỉ phổ hình chữ nhật/sinc.
    5. Hệ thống nhiễu hỗn hợp: Nhiễu Poisson (Shot noise) + Nhiễu Gauss (Read noise).
    """
    y = np.arange(H)
    x = np.arange(W)
    X, Y = np.meshgrid(x, y)
    
    # 1. Mô phỏng biên độ vật thể không đều (Vùng pha dày sẽ hấp thụ bớt ánh sáng)
    # Tế bào thực tế có độ hấp thụ nhẹ khoảng 5% - 20% ở vùng nhân dày
    A_obj = 1.0 - 0.15 * (phi - phi.min()) / (phi.max() - phi.min() + 1e-8)
    U_obj_raw = A_obj * np.exp(1j * phi)
    
    # 2. Mô phỏng giới hạn độ phân giải của Vật kính (Microscope Objective Pupil)
    # Lọc thông thấp trường sóng vật thể trong miền Fourier trước khi cho giao thoa
    U_obj_fft = np.fft.fftshift(np.fft.fft2(U_obj_raw))
    y_grid, x_grid = np.ogrid[-H//2:H//2, -W//2:W//2]
    dist_fft = np.sqrt(x_grid**2 + y_grid**2)
    pupil_mask = (dist_fft <= objective_na_radius).astype(np.float32)
    U_obj_fft_filtered = U_obj_fft * pupil_mask
    U_obj = np.fft.ifft2(np.fft.ifftshift(U_obj_fft_filtered)) # Trường sóng vật thể thực tế bị giới hạn quang học
    
    # 3. Hồ sơ cường độ chùm sáng Gaussian (Gaussian Beam Profile)
    # Ánh sáng laser thực tế sáng ở tâm và mờ dần ra rìa ảnh
    sigma_beam = 0.7 * min(H, W)
    A_beam = np.exp(-((X - W/2)**2 + (Y - H/2)**2) / (2 * sigma_beam**2))
    
    # 4. Sóng mang sub-pixel ngẫu nhiên (Sub-pixel Carrier Frequencies)
    kx1 = rng.uniform(35.2, 49.8) # Số thập phân gây rò rỉ phổ thực tế
    ky1 = rng.uniform(-44.8, -30.2)
    
    kx2 = rng.uniform(-49.8, -35.2)
    ky2 = rng.uniform(-44.8, -30.2)
    
    # 5. Sinh sóng tham chiếu phức có biên độ Gaussian
    phase_carrier1 = 2 * np.pi * (kx1 * X / W + ky1 * Y / H)
    phase_carrier2 = 2 * np.pi * (kx2 * X / W + ky2 * Y / H)
    
    # Sóng tham chiếu cũng đi qua profile chùm sáng Gaussian
    R1 = A_beam * np.exp(1j * phase_carrier1)
    R2 = A_beam * np.exp(1j * phase_carrier2)
    
    # Nhân trường sóng vật thể với biên độ chùm sáng
    U_obj_beam = U_obj * A_beam
    
    # 6. Giao thoa vật lý tạo hologram cường độ
    I1 = np.abs(U_obj_beam + R1)**2
    I2 = np.abs(U_obj_beam + R2)**2
    
    # 7. Mô phỏng nhiễu hỗn hợp thực tế:
    # a) Nhiễu Poisson (Shot noise): Sai số tỉ lệ thuận với căn bậc hai của cường độ
    shot_noise_scale = 0.015
    I1 += rng.normal(0, shot_noise_scale * np.sqrt(np.clip(I1, 0, None)), size=I1.shape)
    I2 += rng.normal(0, shot_noise_scale * np.sqrt(np.clip(I2, 0, None)), size=I2.shape)
    
    # b) Nhiễu đọc Gauss (Read noise): Nhiễu nền camera cố định
    I1 += rng.normal(0, noise_level, size=I1.shape)
    I2 += rng.normal(0, noise_level, size=I2.shape)
    
    # Chuẩn hóa ảnh về dải [0, 1] như ảnh camera ghi nhận
    I1 = (I1 - I1.min()) / (I1.max() - I1.min() + 1e-8)
    I2 = (I2 - I2.min()) / (I2.max() - I2.min() + 1e-8)
    
    return I1.astype(np.float32), I2.astype(np.float32), (kx1, ky1), (kx2, ky2)


class MultiAngleHologramDataset(Dataset):
    """
    Bộ dữ liệu PyTorch hỗ trợ đồng thời sinh dữ liệu mô phỏng tiệm cận thực nghiệm
    và đọc ảnh thực tế (.bmp, .tif, .png) có tự động ước lượng sóng mang.
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
            self.sample_seeds = self.rng.integers(0, 2**32 - 1, size=num_samples)
        else:
            if not data_dir or not os.path.exists(data_dir):
                raise ValueError(f"Đường dẫn dữ liệu thực tế không hợp lệ: {data_dir}")
            self.pairs = self._find_real_pairs(data_dir)
            if len(self.pairs) == 0:
                print(f"⚠️ Cảnh báo: Không tìm thấy cặp ảnh hologram phù hợp trong {data_dir}!")
            self.num_samples = len(self.pairs)
            
    def _find_real_pairs(self, data_dir):
        extensions = ['*.bmp', '*.tif', '*.tiff', '*.png']
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(data_dir, ext)))
            all_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
            
        all_files = sorted(list(set(all_files)))
        pairs = []
        
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
                
        for prefix, angles in prefix_dict.items():
            if 1 in angles and 2 in angles:
                pairs.append((angles[1], angles[2]))
                
        if len(pairs) == 0 and len(all_files) >= 2:
            print("💡 Không tìm thấy hậu tố ghép cặp. Ghép cặp tuần tự các file ảnh...")
            for i in range(0, len(all_files) - 1, 2):
                pairs.append((all_files[i], all_files[i+1]))
                
        return pairs
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        if self.mode == 'synthetic':
            seed = int(self.sample_seeds[idx])
            local_rng = np.random.default_rng(seed)
            
            phi = generate_synthetic_phase_cell(self.H, self.W, local_rng)
            
            # Sử dụng hàm sinh hologram thực tế (realistic simulation)
            I1, I2, (kx1, ky1), (kx2, ky2) = generate_hologram_pair_realistic(phi, self.H, self.W, local_rng)
            
            I1_tensor = torch.from_numpy(I1).unsqueeze(0)
            I2_tensor = torch.from_numpy(I2).unsqueeze(0)
            phi_tensor = torch.from_numpy(phi).unsqueeze(0)
            
            k1_tensor = torch.tensor([kx1, ky1], dtype=torch.float32)
            k2_tensor = torch.tensor([kx2, ky2], dtype=torch.float32)
            
            return {
                'I1': I1_tensor,
                'I2': I2_tensor,
                'k1': k1_tensor,
                'k2': k2_tensor,
                'phi_gt': phi_tensor
            }
        else:
            img1_path, img2_path = self.pairs[idx]
            
            I1_raw = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
            I2_raw = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
            
            if I1_raw.shape != (self.H, self.W):
                I1_raw = cv2.resize(I1_raw, (self.W, self.H), interpolation=cv2.INTER_AREA)
            if I2_raw.shape != (self.H, self.W):
                I2_raw = cv2.resize(I2_raw, (self.W, self.H), interpolation=cv2.INTER_AREA)
                
            I1 = I1_raw.astype(np.float32)
            I2 = I2_raw.astype(np.float32)
            
            max_val1 = 65535.0 if I1_raw.dtype == np.uint16 else 255.0
            max_val2 = 65535.0 if I2_raw.dtype == np.uint16 else 255.0
            
            I1 /= max_val1
            I2 /= max_val2
            
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
                'phi_gt': torch.zeros_like(I1_tensor)
            }

if __name__ == "__main__":
    print("⏳ Đang kiểm tra chức năng sinh dữ liệu mô phỏng TIỆM CẬN THỰC NGHIỆM...")
    dataset = MultiAngleHologramDataset(mode='synthetic', num_samples=5, image_size=(256, 256))
    sample = dataset[0]
    
    print("✅ Kiểm tra thành công!")
    print(f"Hologram 1 shape: {sample['I1'].shape}, Dải giá trị: [{sample['I1'].min():.2f}, {sample['I1'].max():.2f}]")
    print(f"Hologram 2 shape: {sample['I2'].shape}")
    print(f"Sóng mang thực tế (Góc lẻ thập phân): {sample['k1'].numpy()}")

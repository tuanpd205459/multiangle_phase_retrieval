import os
import glob
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def _blank_dc_region(fft_amp, dc_radius=None, dc_stripe_width=10):
    """
    Xóa vùng bậc 0 (DC / zero-order) khỏi phổ biên độ FFT trước khi phân tích.

    Hai thành phần cần xóa:
      1. Vùng tròn quanh tâm (cx, cy) bán kính dc_radius:
         Chứa toàn bộ năng lượng bậc 0 (DC + object DC + background).
      2. Dải đứng |x - cx| < dc_stripe_width:
         Do rò rỉ phổ (spectral leakage) của DC theo trục X khi cửa sổ không tuần hoàn.

    Trả về fft_amp đã xóa vùng DC (bản copy, không thay đổi input).
    """
    H, W = fft_amp.shape
    cx, cy = W // 2, H // 2

    if dc_radius is None:
        # Mặc định: loại bỏ 12% chiều rộng ảnh xung quanh tâm
        dc_radius = max(int(min(H, W) * 0.12), 20)

    y_coords = np.arange(H)
    x_coords = np.arange(W)
    X, Y = np.meshgrid(x_coords, y_coords)

    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Mask vùng cần xóa: hình tròn DC + dải đứng
    dc_circle  = dist_from_center <= dc_radius
    dc_stripe  = np.abs(X - cx) < dc_stripe_width
    dc_region  = dc_circle | dc_stripe

    result = fft_amp.copy()
    result[dc_region] = 0.0
    return result

def estimate_carrier_frequency(I, search_radius_min=28, search_radius_max=90, thresh_ratio=0.85):
    """
    Ước lượng tự động tần số sóng mang (kx, ky) dạng sub-pixel của hologram
    bằng phương pháp làm mịn phổ (Gaussian smoothing) và tính Trọng tâm Năng lượng.
    Trước khi tìm kiếm, toàn bộ vùng bậc 0 (DC + dải đứng) bị xóa khỏi phổ
    bằng _blank_dc_region() để tránh nhầm bậc +1 với DC leakage.
    """
    import scipy.ndimage as ndimage
    H, W = I.shape
    I_fft = np.fft.fftshift(np.fft.fft2(I))
    I_fft_amp = np.abs(I_fft)

    # 1. Xóa hoàn toàn vùng DC trước mọi phân tích
    I_fft_amp_no_dc = _blank_dc_region(I_fft_amp,
                                       dc_radius=search_radius_min,
                                       dc_stripe_width=10)

    # 2. Làm mịn phổ đã xóa DC bằng Gaussian để làm nổi bật búp phổ +1
    I_fft_amp_smooth = ndimage.gaussian_filter(I_fft_amp_no_dc, sigma=5.0)

    # 3. Tạo lưới tọa độ pixel 2D
    y_coords = np.arange(H)
    x_coords = np.arange(W)
    X, Y = np.meshgrid(x_coords, y_coords)

    # 4. Vùng tìm kiếm: annulus [search_radius_min, search_radius_max], nửa bên phải
    #    (Không cần loại thêm DC ở đây vì đã xóa rồi)
    dist_from_dc = np.sqrt((X - W//2)**2 + (Y - H//2)**2)
    search_mask = (
        (dist_from_dc >= search_radius_min) &
        (dist_from_dc <= search_radius_max) &
        (X - W//2 >= 0)   # chỉ tìm nửa phải (kx > 0)
    )

    masked_smooth = I_fft_amp_smooth * search_mask

    # 5. Tìm đỉnh trong vùng tìm kiếm
    max_val = np.max(masked_smooth)
    if max_val == 0:
        return 0.0, 0.0

    # 6. Ngưỡng năng lượng để tính trọng tâm
    threshold = thresh_ratio * max_val
    high_energy_mask = (masked_smooth >= threshold) & search_mask

    # 7. Tính trọng tâm (Center of Mass) dựa trên phổ gốc (không smooth) để giữ độ chính xác
    weights = I_fft_amp_no_dc[high_energy_mask]
    total_weight = np.sum(weights)

    if total_weight == 0:
        max_idx = np.argmax(masked_smooth)
        peak_y, peak_x = np.unravel_index(max_idx, masked_smooth.shape)
        ky = float(peak_y - H//2)
        kx = float(peak_x - W//2)
    else:
        centroid_y = np.sum(Y[high_energy_mask] * weights) / total_weight
        centroid_x = np.sum(X[high_energy_mask] * weights) / total_weight
        ky = float(centroid_y - H//2)
        kx = float(centroid_x - W//2)

    return kx, ky

def estimate_filter_size(I, kx, ky, energy_thresh=0.15, min_rx=15.0, min_ry=15.0, margin=5.0):
    """
    Ước lượng kích thước bộ lọc HCN cần thiết để bao phủ toàn bộ búp phổ +1,
    đồng thời đảm bảo KHÔNG lấn vào vùng phổ bậc 0 (DC).

    Pipeline:
      1. Nhân hologram với exp(-i*2pi*(kx*x/W + ky*y/H)) → dịch bậc +1 về tâm trong FFT.
         Hệ quả: bậc 0 (DC) bị dịch đến (-kx, -ky) so với tâm.
      2. Đo vùng có năng lượng > energy_thresh * max trong bán kính lobe_search_r quanh tâm,
         loại trừ vùng lân cận vị trí DC sau shift (tại (-kx, -ky)).
      3. Tính rx_est, ry_est từ vùng đo được.
      4. Hard-cap rx < abs(kx) - dc_gap_x để cạnh hộp không chồng lên DC khi visualize.

    Trả về:
      rx (float): Bán kính bộ lọc theo trục X (half-width), không chồng DC
      ry (float): Bán kính bộ lọc theo trục Y (half-height)
    """
    H, W = I.shape
    y = np.arange(H)
    x = np.arange(W)
    X, Y = np.meshgrid(x, y)

    # 1. FFT trực tiếp (KHÔNG phase-shift) → bậc 0 ở tâm, bậc +1 ở (cx+kx, cy+ky)
    I_fft = np.fft.fftshift(np.fft.fft2(I))
    amp = np.abs(I_fft)

    # 2. Xóa vùng DC trước mọi phân tích
    amp_no_dc = _blank_dc_region(amp,
                                  dc_radius=int(max(abs(kx), abs(ky)) * 0.5),
                                  dc_stripe_width=10)

    cx, cy = W // 2, H // 2

    # 3. Vùng tìm lobe +1: hình tròn quanh đỉnh sóng mang (cx+kx, cy+ky)
    peak_cx = cx + kx
    peak_cy = cy + ky
    dist_from_peak = np.sqrt((X - peak_cx)**2 + (Y - peak_cy)**2)
    lobe_search_r = max(abs(kx) * 0.6, 20.0)
    lobe_mask = dist_from_peak <= lobe_search_r

    amp_in_lobe = amp_no_dc * lobe_mask
    max_val = amp_in_lobe.max()
    if max_val == 0:
        return max(min_rx, abs(kx) * 0.5), min_ry

    # 4. Ngưỡng năng lượng để xác định vùng búp phổ
    threshold = energy_thresh * max_val
    lobe_region = amp_in_lobe >= threshold

    if lobe_region.sum() == 0:
        return max(min_rx, abs(kx) * 0.5), min_ry

    # 5. Đo half-width và half-height của vùng lobe (so với đỉnh sóng mang)
    lobe_xs = X[lobe_region] - peak_cx
    lobe_ys = Y[lobe_region] - peak_cy

    rx_est = float(np.max(np.abs(lobe_xs))) + margin
    ry_est = float(np.max(np.abs(lobe_ys))) + margin

    # 6. Hard-cap rx để cạnh hộp KHÔNG chồng lên DC (bậc 0 tại cx trong FFT gốc)
    #    Cạnh trái hộp = peak_cx - rx = cx + kx - rx phải > cx → rx < kx - dc_gap_x
    dc_gap_x = 12.0
    max_rx_safe = max(abs(kx) - dc_gap_x, min_rx)
    rx = min(max(rx_est, min_rx), max_rx_safe)

    if abs(ky) > 10.0:
        dc_gap_y = 12.0
        max_ry_safe = max(abs(ky) - dc_gap_y, min_ry)
        ry = min(max(ry_est, min_ry), max_ry_safe)
    else:
        ry = max(ry_est, min_ry)

    return rx, ry


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
                 image_size=(256, 256), seed=42, transform=None, is_eval=False):
        self.mode = mode.lower()
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.H, self.W = image_size
        self.transform = transform
        self.is_eval = is_eval
        
        if self.mode == 'synthetic':
            self.rng = np.random.default_rng(seed)
            self.sample_seeds = self.rng.integers(0, 2**32 - 1, size=num_samples)
        else:
            if not data_dir or not os.path.exists(data_dir):
                raise ValueError(f"Đường dẫn dữ liệu thực tế không hợp lệ: {data_dir}")
            self.groups = self._find_real_groups(data_dir)
            if len(self.groups) == 0:
                print(f"⚠️ Cảnh báo: Không tìm thấy nhóm ảnh hologram phù hợp trong {data_dir}!")
            self.num_samples = len(self.groups)
            
    def _find_real_groups(self, data_dir):
        """
        Tìm và gom nhóm tất cả các ảnh có cùng tiền tố mẫu nhưng khác số thứ tự góc chiếu.
        Ví dụ: 'sample_001_1.bmp', 'sample_001_2.bmp', 'sample_001_3.bmp' -> 1 nhóm 3 ảnh.
        """
        extensions = ['*.bmp', '*.tif', '*.tiff', '*.png']
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(data_dir, ext)))
            all_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
            
        all_files = sorted(list(set(all_files)))
        groups = []
        
        # Biểu thức tìm kiếm: nhóm các góc bằng số nguyên ở cuối, hỗ trợ cả dấu ngoặc đơn dạng im3 (1).bmp
        pattern = re.compile(r'^(.*?)(?:_angle|_goc|_|\s\(|\()([0-9]+)\)?\.[a-zA-Z0-9]+$')
        
        prefix_dict = {}
        for fpath in all_files:
            fname = os.path.basename(fpath)
            match = pattern.match(fname)
            if match:
                prefix = match.group(1)
                angle = int(match.group(2))
                if prefix not in prefix_dict:
                    prefix_dict[prefix] = []
                prefix_dict[prefix].append((angle, fpath))
                
        for prefix, files in prefix_dict.items():
            if len(files) >= 2:
                # Sắp xếp các ảnh trong nhóm theo thứ tự số góc chiếu tăng dần
                files_sorted = [fpath for angle, fpath in sorted(files, key=lambda x: x[0])]
                groups.append(files_sorted)
                
        if len(groups) == 0 and len(all_files) >= 2:
            print("💡 Không tìm thấy hậu tố góc dạng số. Ghép cặp tuần tự các file ảnh...")
            for i in range(0, len(all_files) - 1, 2):
                groups.append([all_files[i], all_files[i+1]])
                
        return groups
        
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
            group = self.groups[idx]
            
            # Chọn 2 ảnh từ nhóm N ảnh
            if self.is_eval:
                # Trong chế độ đánh giá, cố định chọn 2 góc đầu tiên để đảm bảo tính nhất quán của kết quả
                idx1, idx2 = 0, 1
            else:
                # Trong chế độ huấn luyện, ngẫu nhiên chọn 2 góc bất kỳ từ N góc của mẫu vật
                idx1, idx2 = sorted(np.random.choice(len(group), size=2, replace=False))
                
            img1_path = group[idx1]
            img2_path = group[idx2]
            
            # Đọc ảnh an toàn với đường dẫn chứa ký tự Unicode trên Windows
            try:
                I1_raw = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                I2_raw = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            except Exception as e:
                I1_raw, I2_raw = None, None

            if I1_raw is None or I2_raw is None:
                raise FileNotFoundError(f"Không thể đọc ảnh từ: {img1_path} hoặc {img2_path}")
            
            # Chuyển đổi sang ảnh xám 1 kênh nếu ảnh thực tế là ảnh màu 3 kênh (RGB/BGR)
            if len(I1_raw.shape) == 3:
                I1_raw = cv2.cvtColor(I1_raw, cv2.COLOR_BGR2GRAY)
            if len(I2_raw.shape) == 3:
                I2_raw = cv2.cvtColor(I2_raw, cv2.COLOR_BGR2GRAY)
            
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

            # Ước lượng kích thước bộ lọc HCN dựa trên kích thước búp phổ +1 thực tế
            rx1, ry1 = estimate_filter_size(I1, kx1, ky1)
            rx2, ry2 = estimate_filter_size(I2, kx2, ky2)

            I1_tensor = torch.from_numpy(I1).unsqueeze(0)
            I2_tensor = torch.from_numpy(I2).unsqueeze(0)
            
            k1_tensor = torch.tensor([kx1, ky1], dtype=torch.float32)
            k2_tensor = torch.tensor([kx2, ky2], dtype=torch.float32)
            filter1_tensor = torch.tensor([rx1, ry1], dtype=torch.float32)
            filter2_tensor = torch.tensor([rx2, ry2], dtype=torch.float32)
            
            return {
                'I1': I1_tensor,
                'I2': I2_tensor,
                'k1': k1_tensor,
                'k2': k2_tensor,
                'filter1': filter1_tensor,  # [rx1, ry1] ước lượng từ phổ thực
                'filter2': filter2_tensor,  # [rx2, ry2] ước lượng từ phổ thực
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

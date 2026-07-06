import os
import glob
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def estimate_carrier_frequency(I, search_radius_min=28, search_radius_max=90, thresh_ratio=0.85):
    """
    Ước lượng tự động tần số sóng mang (kx, ky) sử dụng thuật toán tránh DC:
      1. Tính phổ Log-amplitude mịn.
      2. Tự động phát hiện vùng DC trong bán kính trung tâm r0=60 và phân ngưỡng Otsu.
      3. Giãn nở vùng DC tạo Forbidden Zone (vùng cấm).
      4. Tìm đỉnh có biên độ lớn nhất ngoài vùng cấm ở nửa bên phải của phổ (kx > 0).
    """
    import scipy.ndimage as ndimage
    H, W = I.shape
    cx, cy = W // 2, H // 2
    
    # Tính phổ Log-amplitude mịn
    I_fft = np.fft.fftshift(np.fft.fft2(I))
    amp = np.abs(I_fft)
    spec = np.log(1.0 + amp + 1e-6)
    spec_smooth = ndimage.gaussian_filter(spec, sigma=2.0)
    
    # Tạo lưới tọa độ
    y_coords = np.arange(H)
    x_coords = np.arange(W)
    X, Y = np.meshgrid(x_coords, y_coords)
    dist_from_dc = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Bắt vùng DC tự động
    r0 = 60
    center_mask = dist_from_dc <= r0
    center_spec = spec_smooth.copy()
    center_spec[~center_mask] = 0
    
    # Chuẩn hóa để chạy ngưỡng Otsu
    c_min = center_spec.min()
    c_max = center_spec.max()
    if c_max - c_min > 1e-8:
        center_spec_gray = (center_spec - c_min) / (c_max - c_min)
        center_spec_u8 = (center_spec_gray * 255).astype(np.uint8)
        _, bw_dc = cv2.threshold(center_spec_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_dc = bw_dc > 0
    else:
        bw_dc = dist_from_dc <= 20
        
    # Giãn nở vùng DC tạo Forbidden Zone để ngăn chặn triệt để lọt phổ vào DC
    margin_dc = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_dc + 1, 2 * margin_dc + 1))
    forbidden_zone = cv2.dilate(bw_dc.astype(np.uint8), kernel) > 0
    
    # Chỉ tìm kiếm đỉnh bậc +1 ở nửa bên phải (X >= cx + 12 để tránh leakage trực tiếp)
    search_space = spec_smooth.copy()
    search_space[forbidden_zone] = 0
    search_space[X < cx + 12] = 0  # Tránh DC cột trung tâm
    
    # Lọc theo annulus khoảng cách của búp phổ
    search_mask = (dist_from_dc >= search_radius_min) & (dist_from_dc <= search_radius_max)
    search_space[~search_mask] = 0
    
    max_val = np.max(search_space)
    if max_val == 0:
        # Fallback nếu không có tín hiệu nào thỏa mãn
        return float(search_radius_min + 10), 0.0
        
    max_idx = np.argmax(search_space)
    py, px = np.unravel_index(max_idx, search_space.shape)
    
    # Tính trọng tâm xung quanh hạt nhân đỉnh này để đạt độ chính xác sub-pixel
    local_r = 7
    local_mask = (np.sqrt((X - px)**2 + (Y - py)**2) <= local_r) & search_mask & ~forbidden_zone
    weights = amp[local_mask]
    total_weight = np.sum(weights)
    
    if total_weight > 0:
        centroid_y = np.sum(Y[local_mask] * weights) / total_weight
        centroid_x = np.sum(X[local_mask] * weights) / total_weight
        ky = float(centroid_y - cy)
        kx = float(centroid_x - cx)
    else:
        ky = float(py - cy)
        kx = float(px - cx)
        
    return kx, ky

def estimate_filter_size(I, kx, ky, energy_thresh=0.15, min_rx=15.0, min_ry=15.0, margin=5.0):
    """
    Ước lượng kích thước bộ lọc (rx, ry) cho búp phổ +1 dựa trên thuật toán loang vùng (Region Growing):
      1. Tạo Forbidden Zone (vùng cấm) quanh DC tương tự.
      2. Loang vùng từ đỉnh sóng mang (cx+kx, cy+ky) sử dụng cv2.floodFill tốc độ cao.
      3. Thực hiện morphology để làm mịn và điền lỗ cho búp phổ thu được.
      4. Tính bounding box rx, ry và cap cứng cạnh trái để tránh đè lên DC.
    """
    import scipy.ndimage as ndimage
    H, W = I.shape
    cx, cy = W // 2, H // 2
    
    # 1. FFT và tính phổ log mịn (dành cho phát hiện DC) và phổ biên độ mịn (dành cho loang búp phổ)
    I_fft = np.fft.fftshift(np.fft.fft2(I))
    amp = np.abs(I_fft)
    spec = np.log(1.0 + amp + 1e-6)
    spec_smooth = ndimage.gaussian_filter(spec, sigma=2.0)
    amp_smooth = ndimage.gaussian_filter(amp, sigma=2.0)
    
    # 2. Tạo Forbidden Zone quanh DC để tránh loang lấn vào DC (dùng spec_smooth)
    y_coords = np.arange(H)
    x_coords = np.arange(W)
    X, Y = np.meshgrid(x_coords, y_coords)
    dist_from_dc = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    r0 = 60
    center_mask = dist_from_dc <= r0
    center_spec = spec_smooth.copy()
    center_spec[~center_mask] = 0
    
    c_min = center_spec.min()
    c_max = center_spec.max()
    if c_max - c_min > 1e-8:
        center_spec_gray = (center_spec - c_min) / (c_max - c_min)
        center_spec_u8 = (center_spec_gray * 255).astype(np.uint8)
        _, bw_dc = cv2.threshold(center_spec_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_dc = bw_dc > 0
    else:
        bw_dc = dist_from_dc <= 20
        
    margin_dc = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_dc + 1, 2 * margin_dc + 1))
    forbidden_zone = cv2.dilate(bw_dc.astype(np.uint8), kernel) > 0
    
    # Vị trí hạt giống (seed)
    seed_x = int(np.clip(cx + kx, 0, W - 1))
    seed_y = int(np.clip(cy + ky, 0, H - 1))
    
    # 3. LOẠI BỎ VÙNG DC VÀ NỬA BÊN TRÁI: Gán năng lượng vùng DC và toàn bộ nửa bên trái về 0
    amp_smooth_no_dc = amp_smooth.copy()
    amp_smooth_no_dc[forbidden_zone] = 0.0
    amp_smooth_no_dc[X < cx + 12] = 0.0  # Chỉ giữ lại nửa bên phải (bậc +1)
    
    # 4. Seeded Region Growing sử dụng cv2.floodFill (C++ optimized) trên phổ đã sạch DC và nửa trái
    peak_amp = amp_smooth_no_dc[seed_y, seed_x]
    
    # Mask cho floodFill cần kích thước (H+2, W+2)
    ff_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    ff_mask[1:-1, 1:-1] = forbidden_zone.astype(np.uint8)
    
    # Ngưỡng dừng loang: dừng khi biên độ giảm xuống 10% biên độ đỉnh
    alpha = 0.10
    lo_diff = float((1.0 - alpha) * peak_amp)
    up_diff = float(999.0 * peak_amp)  # cho phép loang vào vùng có biên độ cao hơn thoải mái
    
    flood_img = amp_smooth_no_dc
    try:
        cv2.floodFill(
            image=flood_img,
            mask=ff_mask,
            seedPoint=(seed_x, seed_y),
            newVal=0,
            loDiff=lo_diff,
            upDiff=up_diff,
            flags=4 | cv2.FLOODFILL_FIXED_RANGE
        )
        mask = ff_mask[1:-1, 1:-1] > 0
    except Exception:
        # Fallback nếu floodFill lỗi
        mask = np.sqrt((X - seed_x)**2 + (Y - seed_y)**2) <= min_rx
        
    # Loại bỏ vùng cấm ra khỏi kết quả loang
    mask[forbidden_zone] = False
    
    # 4. Hậu xử lý Morphology: imclose và điền lỗ
    mask_u8 = mask.astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    
    # Điền lỗ (fill holes)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask_u8)
    if len(contours) > 0:
        cv2.drawContours(mask_filled, contours, -1, 255, -1)
    else:
        mask_filled = mask_u8
        
    # 5. Xác định bounding box rx, ry từ mask
    y_idx, x_idx = np.where(mask_filled > 0)
    if len(x_idx) > 0:
        rx_est = float(np.max(np.abs(x_idx - seed_x))) + margin
        ry_est = float(np.max(np.abs(y_idx - seed_y))) + margin
    else:
        rx_est, ry_est = min_rx, min_ry
        
    # 6. Ràng buộc an toàn tuyệt đối tránh đè lên DC
    # Cạnh trái của bộ lọc (seed_x - rx) phải cách DC (cx) một khoảng an toàn (dc_gap_x = 12px)
    dc_gap_x = 12.0
    max_rx_safe = max(abs(kx) - dc_gap_x, min_rx)
    rx = min(max(rx_est, min_rx), max_rx_safe)
    
    # Ràng buộc ky tương tự
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

            # 1. Ước lượng kích thước bộ lọc độc lập cho từng góc
            rx1_est, ry1_est = estimate_filter_size(I1, kx1, ky1)
            rx2_est, ry2_est = estimate_filter_size(I2, kx2, ky2)

            # 2. Đồng bộ kích thước bộ lọc chung (lấy max của cả hai góc)
            # Vì cùng mẫu vật và hệ thống quang học thì búp phổ phải có kích thước vật lý như nhau
            rx_shared = max(rx1_est, rx2_est)
            ry_shared = max(ry1_est, ry2_est)

            # 3. Áp dụng giới hạn an toàn tránh DC riêng biệt cho từng góc dựa trên kx, ky của nó
            dc_gap_x = 12.0
            min_rx = 15.0
            min_ry = 15.0
            
            # An toàn cho góc 1
            max_rx1_safe = max(abs(kx1) - dc_gap_x, min_rx)
            rx1 = min(rx_shared, max_rx1_safe)
            if abs(ky1) > 10.0:
                max_ry1_safe = max(abs(ky1) - dc_gap_x, min_ry)
                ry1 = min(ry_shared, max_ry1_safe)
            else:
                ry1 = ry_shared

            # An toàn cho góc 2
            max_rx2_safe = max(abs(kx2) - dc_gap_x, min_rx)
            rx2 = min(rx_shared, max_rx2_safe)
            if abs(ky2) > 10.0:
                max_ry2_safe = max(abs(ky2) - dc_gap_x, min_ry)
                ry2 = min(ry_shared, max_ry2_safe)
            else:
                ry2 = ry_shared

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

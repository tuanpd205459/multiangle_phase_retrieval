import os
import glob
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def fourier_region_recognition(I, min_area=30, margin=5.0):
    """
    Thuật toán nhận diện vùng tự động trên miền Fourier theo bài báo:
    'Automated Fourier space region-recognition filtering for off-axis digital holographic microscopy'
    
    Quy trình (đúng 100% theo paper):
      (1) Apply GTL (Otsu) to intensity of FFT hologram → binary image → regionprops.
      (2) Increase threshold by 1% of GTL, repeat until number of regions == 3.
      (3) Use box boundary data to get the right frequency component boundary as filtering window.
    """
    import scipy.ndimage as ndimage
    H, W = I.shape
    cx, cy = W // 2, H // 2
    
    # 1. Biến đổi Fourier và tính biên độ (RAW amplitude, KHÔNG dùng log)
    I_fft = np.fft.fftshift(np.fft.fft2(I))
    amp = np.abs(I_fft)
    
    # 2. Tính ngưỡng Otsu khởi đầu (GTL) trên RAW amplitude
    #    graythresh trong MATLAB hoạt động trên dữ liệu float trực tiếp.
    #    Ở đây ta tự implement Otsu trên float để tránh mất thông tin do uint8.
    amp_flat = amp.ravel()
    nbins = 256
    hist, bin_edges = np.histogram(amp_flat, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    total = hist.sum()
    sum_total = np.sum(bin_centers * hist)
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best_threshold = bin_centers[0]
    
    for i in range(nbins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += bin_centers[i] * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = bin_centers[i]
    
    gtl = best_threshold
    
    T = gtl
    step = 0.01 * gtl
    max_iter = 200
    best_components = None
    num_labels_prev = 0
    step1_binary = None  # Lưu ảnh nhị phân bước 1 (tại ngưỡng GTL)
    
    # Kernel morphology
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Hàm nội bộ: phân ngưỡng + morphology + regionprops
    def _threshold_and_regionprops(threshold):
        bw = (amp > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bw_area = np.zeros_like(bw)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(bw_area, [cnt], -1, 255, -1)
        bw_close = cv2.morphologyEx(bw_area, cv2.MORPH_CLOSE, kernel_close)
        contours_fill, _ = cv2.findContours(bw_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bw_filled = np.zeros_like(bw_close)
        cv2.drawContours(bw_filled, contours_fill, -1, 255, -1)
        bw_open = cv2.morphologyEx(bw_filled, cv2.MORPH_OPEN, kernel_open)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_open)
        comps = []
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                comps.append({
                    'centroid': centroids[label_idx].copy(),
                    'bbox': (stats[label_idx, cv2.CC_STAT_LEFT],
                             stats[label_idx, cv2.CC_STAT_TOP],
                             stats[label_idx, cv2.CC_STAT_WIDTH],
                             stats[label_idx, cv2.CC_STAT_HEIGHT]),
                    'area': int(area)
                })
        return bw_open, comps
    
    # BƯỚC 1: Ảnh nhị phân tại ngưỡng GTL ban đầu (lưu lại cho bước 3)
    step1_binary, step1_components = _threshold_and_regionprops(gtl)
    
    # BƯỚC 2: Tăng ngưỡng 1% GTL cho đến khi số vùng bằng 3
    for _ in range(max_iter):
        _, components = _threshold_and_regionprops(T)
                
        # Điều kiện dừng: đúng 3 vùng (DC, +1, -1)
        if len(components) == 3:
            best_components = components
            break
        elif len(components) == 2 and (best_components is None or num_labels_prev != 3):
            best_components = components
        elif len(components) == 1 and best_components is None:
            best_components = components
            
        num_labels_prev = len(components)
        T += step
        if T >= amp.max():
            break
            
    if best_components is None:
        best_components = components if len(components) > 0 else []
        
    # Phân loại từ bước 2: DC là vùng gần tâm hình học nhất
    if len(best_components) > 0:
        dc_comp = min(best_components, key=lambda c: (c['centroid'][0] - cx)**2 + (c['centroid'][1] - cy)**2)
        sidebands = [c for c in best_components if c is not dc_comp]
    else:
        sidebands = []
        
    if len(sidebands) > 0:
        # Chọn búp ở nửa NỬA TRÊN (centroid_y < cy)
        # Búp chính (+1 order) thường nằm ở nửa trên của phổ nếu góc nghiêng quay quanh trục
        top_sidebands = [c for c in sidebands if c['centroid'][1] < cy]
        if len(top_sidebands) > 0:
            target_comp = max(top_sidebands, key=lambda c: c['area'])
        else:
            target_comp = min(sidebands, key=lambda c: c['centroid'][1])
        
        # Centroid búp từ bước 2 (dùng để định vị)
        target_cx, target_cy = target_comp['centroid']
        
        # BƯỚC 3: Quay lại ảnh nhị phân bước 1 → tìm vùng chứa centroid búp +1
        #          → lấy bounding box LỚN từ bước 1 (đúng theo paper)
        best_step1_dist = float('inf')
        best_step1_comp = None
        for c in step1_components:
            d = np.sqrt((c['centroid'][0] - target_cx)**2 + (c['centroid'][1] - target_cy)**2)
            if d < best_step1_dist:
                best_step1_dist = d
                best_step1_comp = c
        
        if best_step1_comp is not None:
            left, top, w, h = best_step1_comp['bbox']
            px, py = best_step1_comp['centroid']
        else:
            left, top, w, h = target_comp['bbox']
            px, py = target_cx, target_cy
        
        rx = w / 2.0 + margin
        ry = h / 2.0 + margin
    else:
        # Fallback: nếu chỉ tìm được 1 vùng (DC), dùng peak search trên raw amp
        search_amp = amp.copy()
        # Xóa vùng DC
        y_coords = np.arange(H)
        x_coords = np.arange(W)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        dist_from_dc = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
        search_amp[dist_from_dc < 15] = 0
        # Chỉ tìm ở nửa TRÊN (ky < 0 tức là y < cy)
        search_amp[cy:, :] = 0
        max_idx = np.argmax(search_amp)
        py_peak, px_peak = np.unravel_index(max_idx, search_amp.shape)
        px, py = float(px_peak), float(py_peak)
        rx, ry = 20.0, 20.0
        
    # Tinh chỉnh sub-pixel bằng weighted centroid
    y_coords = np.arange(H)
    x_coords = np.arange(W)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    local_r = 7
    local_mask = (np.sqrt((X - px)**2 + (Y - py)**2) <= local_r)
    weights = amp[local_mask]
    total_weight = np.sum(weights)
    
    if total_weight > 0:
        kx = float(np.sum(X[local_mask] * weights) / total_weight - cx)
        ky = float(np.sum(Y[local_mask] * weights) / total_weight - cy)
    else:
        kx = float(px - cx)
        ky = float(py - cy)
        
    return kx, ky, rx, ry


def estimate_carrier_frequency(I, search_radius_min=28, search_radius_max=120, min_area=30):
    """
    Ước lượng tự động tần số sóng mang (kx, ky) dạng sub-pixel sử dụng thuật toán bài báo.
    """
    kx, ky, _, _ = fourier_region_recognition(I, min_area=min_area, margin=5.0)
    return kx, ky


def estimate_filter_size(I, kx, ky, min_area=30, min_rx=15.0, min_ry=15.0, margin=5.0):
    """
    Ước lượng kích thước bộ lọc thích nghi dựa trên thuật toán bài báo.
    Sử dụng box boundary data từ regionprops + Gaussian smoothing biên.
    """
    import scipy.ndimage as ndimage
    H, W = I.shape
    cx, cy = W // 2, H // 2
    
    # Lấy thông số kích thước búp phổ từ thuật toán nhận diện vùng
    _, _, rx_est, ry_est = fourier_region_recognition(I, min_area=min_area, margin=margin)
    
    # Ràng buộc an toàn tránh đè lên DC
    dc_gap_x = 12.0
    max_rx_safe = max(abs(kx) - dc_gap_x, min_rx)
    rx = min(max(rx_est, min_rx), max_rx_safe)
    
    if abs(ky) > 10.0:
        dc_gap_y = 12.0
        max_ry_safe = max(abs(ky) - dc_gap_y, min_ry)
        ry = min(max(ry_est, min_ry), max_ry_safe)
    else:
        ry = max(ry_est, min_ry)
        
    # Tạo mặt nạ hình chữ nhật tại tâm (đúng theo paper: box boundary as filtering window)
    # rồi áp dụng Gaussian smoothing biên
    mask = np.zeros((H, W), dtype=np.float32)
    x1 = max(0, int(cx - rx))
    x2 = min(W, int(cx + rx))
    y1 = max(0, int(cy - ry))
    y2 = min(H, int(cy + ry))
    mask[y1:y2, x1:x2] = 1.0
    
    # Gaussian smoothing biên cửa sổ lọc (đúng theo paper)
    mask_centered = ndimage.gaussian_filter(mask, sigma=2.0)
    mask_centered = mask_centered / (mask_centered.max() + 1e-8)
    
    # Tạo sideband_mask để vẽ contour đối chứng trên phổ
    shift_back_x = int(round(kx))
    shift_back_y = int(round(ky))
    sideband_mask = np.roll(mask, shift_back_y, axis=0)
    sideband_mask = np.roll(sideband_mask, shift_back_x, axis=1)
    
    return rx, ry, mask_centered, sideband_mask


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
    # ÉP BUỘC: Cả kx1 và kx2 ĐỀU PHẢI ÂM (trái) để khi trích xuất búp trái, 
    # cả 2 góc đều ra pha thật (tránh hiện tượng 1 góc ra pha thật, 1 góc ra pha liên hợp)
    kx1 = rng.uniform(-49.8, -35.2) 
    kx2 = rng.uniform(-49.8, -35.2)
    
    # ky1 và ky2 có thể âm hoặc dương tự do, trải đều góc phần tư I và IV (hoặc II và III)
    ky1 = rng.uniform(-45.0, 45.0)
    ky2 = rng.uniform(-45.0, 45.0)
    
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

            # 1. Ước lượng kích thước bộ lọc độc lập cho từng góc và lấy mặt nạ thích nghi mềm dạng 2D
            rx1_est, ry1_est, mask1_centered, _ = estimate_filter_size(I1, kx1, ky1)
            rx2_est, ry2_est, mask2_centered, _ = estimate_filter_size(I2, kx2, ky2)

            # 2. Đồng bộ kích thước bộ lọc chung (lấy max của cả hai góc)
            rx_shared = max(rx1_est, rx2_est)
            ry_shared = max(ry1_est, ry2_est)

            # 3. Áp dụng giới hạn an toàn tránh DC riêng biệt cho từng góc
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
            
            # Chuyển đổi các mặt nạ 2D thích nghi sang Tensor dạng PyTorch [1, H, W]
            mask1_tensor = torch.from_numpy(mask1_centered).unsqueeze(0).to(torch.float32)
            mask2_tensor = torch.from_numpy(mask2_centered).unsqueeze(0).to(torch.float32)
            
            return {
                'I1': I1_tensor,
                'I2': I2_tensor,
                'k1': k1_tensor,
                'k2': k2_tensor,
                'filter1': filter1_tensor,  # [rx1, ry1]
                'filter2': filter2_tensor,  # [rx2, ry2]
                'mask1': mask1_tensor,      # Bộ lọc thích nghi 2D cho góc 1 (Centered)
                'mask2': mask2_tensor,      # Bộ lọc thích nghi 2D cho góc 2 (Centered)
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

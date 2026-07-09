# Hướng Dẫn Từng Bước Chạy Dự Án Khôi Phục Pha Trên Google Colab

Tài liệu này hướng dẫn chi tiết cách chạy huấn luyện và đánh giá mô hình khôi phục pha tự giám sát đa góc trên môi trường Google Colab sử dụng GPU miễn phí.

---

## Bước 1: Đẩy mã nguồn lên GitHub (Thực hiện trên máy tính cá nhân)

Để đưa mã nguồn từ máy cá nhân lên Colab một cách nhanh nhất, anh hãy đẩy code lên một repository GitHub:

1. Mở Git Bash hoặc PowerShell tại thư mục `multiangle_phase_retrieval`:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Multi-angle Phase Retrieval"
   ```
2. Tạo một repository mới trên GitHub (ví dụ đặt tên là `multiangle_phase_retrieval`), chọn chế độ **Private** (Riêng tư) nếu anh muốn bảo mật nghiên cứu.
3. Liên kết và đẩy code lên:
   ```bash
   git remote add origin https://github.com/username/multiangle_phase_retrieval.git
   git branch -M main
   git push -u origin main
   ```
   *(Thay thế `username` bằng tài khoản GitHub thực tế của anh).*

---

## Bước 2: Thiết lập môi trường Google Colab

1. Truy cập vào đường dẫn: [Google Colab](https://colab.research.google.com/).
2. Chọn **New Notebook** (Sổ tay mới).
3. Đổi tên notebook thành `multiangle_phase_retrieval_colab.ipynb`.
4. **Kích hoạt GPU (Cực kỳ quan trọng):**
   * Vào menu **Runtime** (Thời gian chạy) $\rightarrow$ **Change runtime type** (Thay đổi loại thời gian chạy).
   * Chọn **T4 GPU** (hoặc L4/A100 nếu dùng bản trả phí) ở mục Hardware accelerator.
   * Nhấn **Save**.

---

## Bước 3: Các câu lệnh thực thi trên Colab (Chạy cell-by-cell)

Hãy tạo và chạy lần lượt các Cell mã nguồn sau trên Google Colab:

### Cell 1: Kết nối với Google Drive để lưu Checkpoint lâu dài
```python
from google.colab import drive
drive.mount('/content/drive')
```
*(Colab sẽ hiển thị yêu cầu xác thực, anh nhấn đồng ý cấp quyền truy cập Drive).*

### Cell 2: Clone dự án từ GitHub về bộ nhớ tạm của Colab
* **Nếu Repo GitHub ở chế độ Công khai (Public):**
  ```bash
  !git clone https://github.com/username/multiangle_phase_retrieval.git
  ```
* **Nếu Repo GitHub ở chế độ Riêng tư (Private - Cần Token xác thực):**
  ```bash
  # Thay <YOUR_GITHUB_TOKEN> bằng Classic Personal Access Token tạo từ tài khoản GitHub của anh
  !git clone https://<YOUR_GITHUB_TOKEN>@github.com/username/multiangle_phase_retrieval.git
  ```

### Cell 3: Chuyển thư mục làm việc và Cài đặt thư viện
```python
%cd multiangle_phase_retrieval
!pip install -r requirements.txt
```

### Cell 4: Tạo liên kết thư mục checkpoints sang Google Drive
Để khi mô hình huấn luyện, các file trọng số `.pth` sẽ tự động được đồng bộ và lưu trữ vĩnh viễn trên Google Drive của anh (không lo bị mất khi Colab ngắt kết nối):
```bash
!mkdir -p "/content/drive/MyDrive/multiangle_checkpoints"
!rm -rf checkpoints
!ln -s "/content/drive/MyDrive/multiangle_checkpoints" checkpoints
```

### Cell 5: Huấn luyện mô hình (Chạy trên GPU)
* **Huấn luyện từ đầu (Pre-train trên dữ liệu mô phỏng):**
  ```bash
  !python scripts/train.py --config configs/base_config.yaml
  ```
* **Huấn luyện tiếp tục từ Checkpoint (Resume - dùng khi Colab bị ngắt kết nối):**
  ```bash
  !python scripts/train.py --config configs/base_config.yaml --resume checkpoints/latest_model.pth
  ```

### Cell 6: Chạy đánh giá và lưu kết quả
Sau khi huấn luyện hoàn tất, chạy cell này để tính toán các chỉ số MSE, PSNR, xuất file ảnh trực quan và file ma trận `.mat` cho MATLAB:
```bash
!python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --num_test 5
```

### Cell 7: Trực quan hóa các bước thô trung gian (NumPy) ngay trên Colab
Để xem trước ảnh pha trung gian bậc +1 sau IFFT và các bộ lọc Fourier thích nghi mà không cần train:
```python
# Chạy tạo ảnh thô trung gian bằng script NumPy
!python C:\Users\admin\.gemini\antigravity\brain\7fda4bc9-0b32-4498-b5bc-55eb4374dce6\scratch\scratch_visualize.py

# Hiển thị ảnh trực tiếp trong Cell Colab
from IPython.display import Image, display
print("🖼️ Ảnh các bước giải điều chế trung gian (Pha thô ở cột cuối cùng):")
display(Image('outputs/intermediate_steps_preview.png'))
```

### Cell 8: Hiển thị toàn bộ kết quả khôi phục pha từ mô hình DNN
Sau khi chạy Cell 6 (suy diễn bằng mô hình DNN), anh chạy cell này để hiển thị trực tiếp tất cả các ảnh kết quả khôi phục (đã tích hợp ảnh pha trung gian ở cột cuối cùng):
```python
import glob
from IPython.display import Image, display

image_paths = sorted(glob.glob('outputs/visual_evaluation_sample_*.png'))
if len(image_paths) == 0:
    print("❌ Không tìm thấy ảnh kết quả. Hãy chắc chắn anh đã chạy Cell 6 thành công!")
else:
    for path in image_paths:
        print(f"👉 Bản đồ khôi phục: {path}")
        display(Image(path))
```

---

## Bước 4: Tải kết quả về máy tính cá nhân

1. Mở thanh quản lý File bên trái của Google Colab.
2. Tìm đến thư mục `multiangle_phase_retrieval/outputs/`:
   * Anh có thể xem trực tiếp các file ảnh hoặc chuột phải chọn **Download**.
   * File ma trận: `reconstructed_sample.mat` chứa các mảng pha trung gian và pha khôi phục tinh chỉnh để chạy MATLAB.
3. Các file trọng số mô hình đã được lưu an toàn trong Google Drive của anh tại thư mục: `My Drive/multiangle_checkpoints/`.

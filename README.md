# Multi-Angle Phase Retrieval in Off-Axis Digital Holography

Dự án này triển khai phương pháp **Khôi phục pha định lượng tự giám sát đa góc (Self-Supervised Multi-Angle Quantitative Phase Imaging)** trong kỹ thuật ghi ảnh Off-axis Digital Holography (DHM) bằng PyTorch.

Phương pháp sử dụng kiến trúc Siamese chung trọng số kết hợp mô hình quang học vật lý khả vi (Differentiable Physics Model), tối ưu hóa đồng thời thông qua Differentiable Physics Loss và Complex Consistency Loss giữa các góc chụp mà không cần dữ liệu nhãn pha thực tế (ground truth phase).

---

## 📂 Cấu trúc thư mục dự án

```
multiangle_phase_retrieval/
│
├── configs/                  # Quản lý cấu hình huấn luyện
│   └── base_config.yaml      # Cấu hình siêu tham số, kích thước ảnh, hệ số loss...
│
├── data/                     # Thư mục chứa dữ liệu
│   ├── raw/                  # Hologram thực tế dưới dạng ảnh .bmp, .tif, .png
│   ├── processed/            # Dữ liệu phục vụ phân tích đối chứng
│   └── synthetic/            # Dữ liệu mô phỏng được sinh tự động bằng Python
│
├── src/                      # Mã nguồn chính của dự án (Core Package)
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset hỗ trợ đọc ảnh thực tế & sinh dữ liệu mô phỏng
│   ├── demodulator.py        # Bộ giải điều chế miền Fourier khả vi (Differentiable Demodulator)
│   ├── losses.py             # Định nghĩa hàm loss (Physics Loss, Complex Consistency Loss)
│   ├── models/               # Thiết kế mạng nơ-ron
│   │   ├── __init__.py
│   │   ├── unet.py           # Phase Refining U-Net
│   │   └── siamese.py        # Mạng Siamese bọc đa góc
│   └── utils/                # Các thư viện bổ trợ vật lý
│       ├── __init__.py
│       ├── phase_ops.py      # Bù lệch pha toàn cục, phase unwrap, tính gradient pha
│       └── io_helpers.py     # Đọc ghi ảnh TIFF 16-bit, lưu pha kết quả dưới dạng .mat cho MATLAB
│
├── scripts/                  # Kịch bản thực thi dòng lệnh
│   ├── train.py              # Huấn luyện mô hình Siamese tự giám sát
│   ├── evaluate.py           # Kiểm tra mô hình trên tập Test và tính MSE, PSNR, SSIM
│   └── visualize.py          # Trực quan hóa và xuất kết quả phục vụ viết bài báo
│
├── checkpoints/              # Nơi lưu trữ trọng số mô hình đã huấn luyện (.pth)
├── outputs/                  # Đầu ra của mô hình (file .mat chứa ma trận pha khôi phục)
├── requirements.txt          # Khai báo thư viện Python cần dùng
└── README.md                 # Tài liệu này
```

---

## 🛠️ Hướng dẫn cài đặt

Dự án yêu cầu cài đặt Python >= 3.10 và driver NVIDIA CUDA thích hợp để huấn luyện trên GPU.

1. **Khởi tạo môi trường ảo (Khuyên dùng):**
   ```bash
   conda create -n multiangle_dh python=3.10 -y
   conda activate multiangle_dh
   ```

2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Hướng dẫn sử dụng

### 1. Cấu hình các tham số huấn luyện
Chỉnh sửa file cấu hình `configs/base_config.yaml` để thay đổi kích thước ảnh, các siêu tham số học tập, trọng số hàm loss hoặc cài đặt tự động lưu checkpoint lên đám mây.

### 2. Sinh dữ liệu giả lập (Thử nghiệm)
Chạy module sinh dữ liệu để tạo tập dữ liệu mô phỏng kiểm thử thuật toán:
```bash
python src/dataset.py
```

### 3. Huấn luyện mô hình tự giám sát
Chạy file train từ kịch bản dòng lệnh:
```bash
python scripts/train.py --config configs/base_config.yaml
```

### 4. Đánh giá và Xuất kết quả sang MATLAB
Đánh giá chất lượng mô hình trên tập kiểm thử độc lập và lưu file kết quả định dạng `.mat` để nạp vào MATLAB:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

---

## ☁️ Quy trình huấn luyện đám mây hỗn hợp (Kaggle & Google Colab)

Để tận dụng tối đa tài nguyên GPU đám mây miễn phí:
1. **GitHub làm trung tâm:** Đẩy (Push) mã nguồn lên repository GitHub của bạn.
2. **Huấn luyện chính trên Kaggle:** 
   * Tạo Notebook mới trên Kaggle, clone repo GitHub về.
   * Tải tập dữ liệu dạng file `.zip` từ Google Drive về Kaggle, giải nén vào ổ cứng tạm của Kaggle.
   * Huấn luyện mô hình. Lưu file checkpoint `.pth` lên Hugging Face Hub (hoặc Google Drive) khi kết thúc phiên.
3. **Tiếp tục và Test trên Colab:**
   * Khi hết quota của Kaggle, mở Google Colab, mount Google Drive hoặc đăng nhập Hugging Face.
   * Tải checkpoint `.pth` gần nhất về và truyền tham số `--resume` vào câu lệnh huấn luyện để chạy tiếp nối.

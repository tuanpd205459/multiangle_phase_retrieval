import os
import sys
import argparse
import yaml

# Thêm thư mục gốc vào sys.path để tránh lỗi import src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset import MultiAngleHologramDataset
from src.utils.dataset_visualizer import save_dataset_preview, save_intermediate_steps_preview

def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa một số mẫu trong dataset và phổ Fourier để kiểm tra")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Đường dẫn đến file cấu hình YAML")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Số lượng mẫu cần trực quan hóa")
    parser.add_argument("--output", type=str, default="outputs/dataset_preview.png",
                        help="Đường dẫn lưu ảnh trực quan hóa")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Lỗi: Không tìm thấy file cấu hình tại {args.config}")
        return
        
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    dataset_mode = 'synthetic' if config['data']['synthetic_data'] else 'real'
    print(f"📦 Đang nạp dữ liệu ở chế độ: {dataset_mode.upper()}")
    
    dataset = MultiAngleHologramDataset(
        mode=dataset_mode,
        data_dir=config['data']['raw_dir'] if dataset_mode == 'real' else None,
        num_samples=args.num_samples,
        image_size=(config['data']['image_height'], config['data']['image_width']),
        seed=config['train']['seed'],
        is_eval=True
    )
    
    print(f"🖼️ Đang tạo ảnh trực quan hóa tập dữ liệu tại: {args.output}")
    save_dataset_preview(
        dataset=dataset,
        output_path=args.output,
        num_samples=args.num_samples,
        filter_radius=config['data']['filter_radius']
    )
    
    steps_output = os.path.join(os.path.dirname(args.output), "intermediate_steps_preview.png")
    print(f"🖼️ Đang tạo ảnh kiểm tra bước trung gian (Fourier demodulation) tại: {steps_output}")
    save_intermediate_steps_preview(
        dataset=dataset,
        output_path=steps_output,
        sample_idx=0,
        filter_radius=config['data']['filter_radius']
    )
    print("🎉 Hoàn tất!")

if __name__ == "__main__":
    main()

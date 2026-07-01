import os
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Trực quan hóa Độc lập ma trận phục hồi pha từ file .mat")
    parser.add_argument("--mat_file", type=str, default="outputs/reconstructed_sample.mat",
                        help="Đường dẫn đến file .mat kết quả")
    parser.add_argument("--save_path", type=str, default="outputs/visual_result.png",
                        help="Đường dẫn lưu ảnh trực quan hóa")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.mat_file):
        print(f"❌ Sai: Không tìm thấy file .mat tại {args.mat_file}")
        return
        
    print(f"📖 Đang nạp dữ liệu từ: {args.mat_file}")
    data = sio.loadmat(args.mat_file)
    
    # Kiểm tra các khóa trong file .mat
    required_keys = ['amplitude1', 'phase1', 'hologram1']
    for key in required_keys:
        if key not in data:
            print(f"❌ Sai: File .mat thiếu dữ liệu khóa quan trọng: '{key}'")
            return
            
    amp = data['amplitude1']
    phase = data['phase1']
    holo = data['hologram1']
    
    has_gt = 'phase_gt' in data
    cols = 4 if has_gt else 3
    
    fig, axes = plt.subplots(1, cols, figsize=(15, 4.5))
    
    # 1. Vẽ Hologram cường độ thô
    axes[0].imshow(holo, cmap='gray')
    axes[0].set_title("Raw Hologram")
    axes[0].axis('off')
    
    # 2. Vẽ Biên độ khôi phục
    im_amp = axes[1].imshow(amp, cmap='jet')
    axes[1].set_title("Reconstructed Amplitude")
    axes[1].axis('off')
    fig.colorbar(im_amp, ax=axes[1])
    
    # 3. Vẽ Pha khôi phục
    im_phase = axes[2].imshow(phase, cmap='jet')
    axes[2].set_title("Reconstructed Phase")
    axes[2].axis('off')
    fig.colorbar(im_phase, ax=axes[2])
    
    # 4. Vẽ Pha Ground Truth nếu có
    if has_gt:
        phase_gt = data['phase_gt']
        im_gt = axes[3].imshow(phase_gt, cmap='jet')
        axes[3].set_title("Ground Truth Phase")
        axes[3].axis('off')
        fig.colorbar(im_gt, ax=axes[3])
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.savefig(args.save_path, dpi=150)
    plt.close()
    
    print(f"🎉 Trực quan hóa thành công! Đã lưu ảnh kết quả tại: {args.save_path}")

if __name__ == "__main__":
    main()

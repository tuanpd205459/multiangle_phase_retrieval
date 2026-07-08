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
    required_keys = ['amplitude1', 'hologram1']
    for key in required_keys:
        if key not in data:
            print(f"❌ Sai: File .mat thiếu dữ liệu khóa quan trọng: '{key}'")
            return
            
    amp = data['amplitude1']
    holo = data['hologram1']
    
    # Lấy pha quấn và pha mở
    phase_wrapped = data.get('phase1_wrapped', data.get('phase1', None))
    phase_unwrapped = data.get('phase1_unwrapped', None)
    
    if phase_wrapped is None:
        print("❌ Sai: Không tìm thấy dữ liệu pha trong file .mat")
        return
        
    phase_rough = data.get('phase_rough1', None)
    has_rough = phase_rough is not None
    has_gt = 'phase_gt' in data
    
    # Tính số cột
    cols = 3
    if has_rough:
        cols += 1
    if has_gt or phase_unwrapped is not None:
        cols += 1
        
    fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4.5))
    
    current_col = 0
    
    # 1. Vẽ Hologram cường độ thô
    axes[current_col].imshow(holo, cmap='gray')
    axes[current_col].set_title("Raw Hologram")
    axes[current_col].axis('off')
    current_col += 1
    
    # 2. Vẽ Pha trung gian (nếu có)
    if has_rough:
        im_rough = axes[current_col].imshow(phase_rough, cmap='jet', vmin=-np.pi, vmax=np.pi)
        axes[current_col].set_title("Intermediate Phase (Wrapped)")
        axes[current_col].axis('off')
        fig.colorbar(im_rough, ax=axes[current_col])
        current_col += 1
        
    # 3. Vẽ Biên độ khôi phục
    im_amp = axes[current_col].imshow(amp, cmap='jet')
    axes[current_col].set_title("Reconstructed Amplitude")
    axes[current_col].axis('off')
    fig.colorbar(im_amp, ax=axes[current_col])
    current_col += 1
    
    # 4. Vẽ Pha quấn khôi phục
    im_phase_wrap = axes[current_col].imshow(phase_wrapped, cmap='jet', vmin=-np.pi, vmax=np.pi)
    axes[current_col].set_title("Reconstructed Wrapped Phase")
    axes[current_col].axis('off')
    fig.colorbar(im_phase_wrap, ax=axes[current_col])
    current_col += 1
    
    # 5. Vẽ Pha mở (hoặc Ground Truth nếu không có pha mở)
    if phase_unwrapped is not None:
        im_phase_unwrap = axes[current_col].imshow(phase_unwrapped, cmap='jet')
        axes[current_col].set_title("Reconstructed Unwrapped Phase")
        axes[current_col].axis('off')
        fig.colorbar(im_phase_unwrap, ax=axes[current_col])
    elif has_gt:
        phase_gt = data['phase_gt']
        im_gt = axes[current_col].imshow(phase_gt, cmap='jet', vmin=-np.pi, vmax=np.pi)
        axes[current_col].set_title("Ground Truth Phase")
        axes[current_col].axis('off')
        fig.colorbar(im_gt, ax=axes[current_col])
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.savefig(args.save_path, dpi=150)
    plt.close()
    
    print(f"🎉 Trực quan hóa thành công! Đã lưu ảnh kết quả tại: {args.save_path}")

if __name__ == "__main__":
    main()

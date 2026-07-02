import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

# Thêm thư mục gốc của dự án vào sys.path để chạy từ bất kỳ đâu không bị lỗi import src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Nạp các module từ thư mục src
from src.dataset import MultiAngleHologramDataset
from src.models.siamese import SiameseTeacherModel
from src.losses import compute_total_loss
from src.utils.dataset_visualizer import save_dataset_preview

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Siamese Khôi phục Pha Tự Giám Sát")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Đường dẫn đến file cấu hình YAML")
    parser.add_argument("--resume", type=str, default="",
                        help="Đường dẫn đến file checkpoint để huấn luyện tiếp nối (Resume)")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train():
    args = parse_args()
    config = load_config(args.config)
    
    # Thiết lập seed để tái lập kết quả
    torch.manual_seed(config['train']['seed'])
    
    # Thiết lập thiết bị chạy (GPU CUDA hoặc CPU)
    device = torch.device(config['cloud']['device'] if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Thiết bị sử dụng huấn luyện: {device}")
    
    # Tạo các thư mục lưu kết quả nếu chưa tồn tại
    checkpoint_dir = config['paths']['checkpoint_dir']
    output_dir = config['paths']['output_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Khởi tạo Dataset và DataLoader
    dataset_mode = 'synthetic' if config['data']['synthetic_data'] else 'real'
    print(f"📦 Đang nạp dữ liệu ở chế độ: {dataset_mode.upper()}")
    
    full_dataset = MultiAngleHologramDataset(
        mode=dataset_mode,
        data_dir=config['data']['raw_dir'] if dataset_mode == 'real' else None,
        num_samples=3300 if dataset_mode == 'synthetic' else 3000, # Bổ sung thêm để cắt Val set
        image_size=(config['data']['image_height'], config['data']['image_width']),
        seed=config['train']['seed']
    )
    
    # Trực quan hóa tập dữ liệu để kiểm tra trước khi training
    preview_path = os.path.join(output_dir, "dataset_preview.png")
    save_dataset_preview(
        dataset=full_dataset,
        output_path=preview_path,
        num_samples=3,
        filter_radius=config['data']['filter_radius']
    )
    
    # Chia tập dữ liệu thành Train và Validation (mặc định 90% / 10%)
    val_size = int(len(full_dataset) * config['train']['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=False
    )
    
    print(f"📊 Tập huấn luyện: {train_size} mẫu | Tập kiểm thử: {val_size} mẫu")
    
    # Tính toán giá trị sóng mang trung bình làm khởi tạo vật lý tốt
    print("📍 Đang tính toán sóng mang khởi tạo trung bình từ dataset...")
    k1_list, k2_list = [], []
    num_init = min(100, len(full_dataset))
    for idx in range(num_init):
        s = full_dataset[idx]
        k1_list.append(s['k1'])
        k2_list.append(s['k2'])
    k1_init = torch.stack(k1_list).mean(dim=0).tolist()
    k2_init = torch.stack(k2_list).mean(dim=0).tolist()
    print(f"   - Khởi tạo k1 (góc 1): {k1_init}")
    print(f"   - Khởi tạo k2 (góc 2): {k2_init}")

    # 2. Khởi tạo Mô hình Siamese Teacher
    model = SiameseTeacherModel(
        filter_radius=config['data']['filter_radius'],
        k1_init=k1_init,
        k2_init=k2_init
    ).to(device)
    
    # 3. Khởi tạo Optimizer
    optimizer = Adam(
        model.parameters(), 
        lr=config['train']['learning_rate'], 
        weight_decay=config['train']['weight_decay']
    )
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 4. Hỗ trợ huấn luyện tiếp nối (Resume training)
    if args.resume:
        if os.path.exists(args.resume):
            print(f"🔄 Đang nạp checkpoint từ: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"➡️ Huấn luyện tiếp tục từ Epoch {start_epoch + 1}...")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy checkpoint tại {args.resume}. Sẽ huấn luyện từ đầu.")

    # Thử nghiệm import TensorBoard để ghi nhật ký trực quan
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))
        print("📈 Đã khởi tạo TensorBoard để ghi nhận tiến trình huấn luyện.")
    except ImportError:
        print("💡 Không tìm thấy TensorBoard. Tiến trình huấn luyện sẽ chỉ ghi ra cửa sổ Console.")
        
    # 5. Vòng lặp huấn luyện chính
    epochs = config['train']['epochs']
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_accum = 0.0
        train_loss_phys = 0.0
        train_loss_cons = 0.0
        train_loss_tv = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Chuyển dữ liệu lên GPU
            I1 = batch['I1'].to(device)
            I2 = batch['I2'].to(device)
            k1 = batch['k1'].to(device)
            k2 = batch['k2'].to(device)
            
            optimizer.zero_grad()
            
            # Chạy mô hình Siamese
            (U1, amp1, phase1), (U2, amp2, phase2) = model(I1, k1, I2, k2)
            
            # Lấy các tham số sóng mang học được hiện tại từ model
            B = I1.shape[0]
            k1_learned = model.k1.unsqueeze(0).expand(B, -1)
            k2_learned = model.k2.unsqueeze(0).expand(B, -1)
            
            # Tính toán hàm Loss sử dụng sóng mang học được
            loss, loss_dict = compute_total_loss(U1, U2, I1, I2, k1_learned, k2_learned, config)
            
            # Lan truyền ngược và tối ưu hóa
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            train_loss_phys += loss_dict['loss_phys']
            train_loss_cons += loss_dict['loss_cons']
            train_loss_tv += loss_dict['loss_tv']
            
        # Tính giá trị loss trung bình của epoch
        num_batches = len(train_loader)
        avg_train_loss = train_loss_accum / num_batches
        avg_train_phys = train_loss_phys / num_batches
        avg_train_cons = train_loss_cons / num_batches
        avg_train_tv = train_loss_tv / num_batches
        
        # 6. Chạy đánh giá trên tập Validation
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                I1 = batch['I1'].to(device)
                I2 = batch['I2'].to(device)
                k1 = batch['k1'].to(device)
                k2 = batch['k2'].to(device)
                
                # Chạy mô hình Siamese
                (U1, _, _), (U2, _, _) = model(I1, k1, I2, k2)
                
                # Sử dụng sóng mang học được để tính Loss đánh giá
                B = I1.shape[0]
                k1_learned = model.k1.unsqueeze(0).expand(B, -1)
                k2_learned = model.k2.unsqueeze(0).expand(B, -1)
                
                _, val_loss_dict = compute_total_loss(U1, U2, I1, I2, k1_learned, k2_learned, config)
                val_loss_accum += val_loss_dict['total_loss']
                
        avg_val_loss = val_loss_accum / len(val_loader)
        
        # Ghi nhật ký vào TensorBoard
        if writer:
            writer.add_scalar("Loss/Train_Total", avg_train_loss, epoch)
            writer.add_scalar("Loss/Train_Physics", avg_train_phys, epoch)
            writer.add_scalar("Loss/Train_Consistency", avg_train_cons, epoch)
            writer.add_scalar("Loss/Train_TV", avg_train_tv, epoch)
            writer.add_scalar("Loss/Val_Total", avg_val_loss, epoch)
            
        # In thông tin tiến trình huấn luyện và các tham số học được
        with torch.no_grad():
            k1_print = model.k1.cpu().numpy()
            k2_print = model.k2.cpu().numpy()
            radius_print = model.demodulator.filter_radius.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {avg_train_loss:.4f} (Phys: {avg_train_phys:.4f}, Cons: {avg_train_cons:.4f}, TV: {avg_train_tv:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f}\n"
              f"   📎 Tham số học được: k1=[{k1_print[0]:.3f}, {k1_print[1]:.3f}] | k2=[{k2_print[0]:.3f}, {k2_print[1]:.3f}] | Filter Radius={radius_print:.3f}")
              
        # 7. Lưu trữ Checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        
        # Lưu checkpoint của epoch mới nhất
        latest_path = os.path.join(checkpoint_dir, config['paths']['latest_model_name'])
        torch.save(checkpoint_data, latest_path)
        
        # Lưu checkpoint tốt nhất (Best model) nếu val loss giảm
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data['best_val_loss'] = best_val_loss
            best_path = os.path.join(checkpoint_dir, config['paths']['best_model_name'])
            torch.save(checkpoint_data, best_path)
            print(f"⭐ Đã lưu model tốt nhất mới với Val Loss: {best_val_loss:.4f}")
            
        # Lưu checkpoint dự phòng định kỳ (Backup interval) cho hybrid workflow
        save_interval = config['cloud'].get('checkpoint_save_interval', 10)
        if (epoch + 1) % save_interval == 0:
            interval_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint_data, interval_path)
            print(f"💾 Đã lưu checkpoint định kỳ tại epoch {epoch+1}")
            
            # Hỗ trợ đẩy lên Hugging Face nếu được cấu hình
            if config['cloud'].get('use_huggingface', False) and config['cloud'].get('hf_repo_id', ''):
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    api.upload_file(
                        path_or_fileobj=latest_path,
                        path_in_repo=f"latest_model.pth",
                        repo_id=config['cloud']['hf_repo_id'],
                        repo_type="model"
                    )
                    print(f"☁️ Tự động upload checkpoint lên Hugging Face Repo: {config['cloud']['hf_repo_id']}")
                except Exception as e:
                    print(f"⚠️ Lỗi đẩy file lên Hugging Face: {str(e)}")

    if writer:
        writer.close()
    print("🎉 Quá trình huấn luyện đã hoàn tất thành công!")

if __name__ == "__main__":
    train()

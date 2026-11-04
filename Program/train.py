
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Program.utils.getDataset import create_dataloaders, get_skeleton_edge_index
from Models.ST_GCN import FallPredictionSTGCN
  
CONFIG = {
    'experiment_name': 'STGCN_V2',
    'train_path': 'URFD/splits_mask/urfd_mask_train_in25_out25_st5.pkl',
    'val_path': 'URFD/splits_mask/urfd_mask_val_in25_out25_st5.pkl',
    'results_dir': 'Coba2_pixel',
    
    # Model Params
    'num_nodes': 17,        
    'in_channels': 2,       
    'hidden_channels': 16,
    #'out_channels': 64,    
    'kernel_size': 3,       
    'window_size': 25,      
    'forecast_window': 25,  
    'num_classes': 3,       
    
    # Training Params
    'epochs': 1000,          
    'batch_size': 64,
    'learning_rate': 1e-4, 
    'weight_decay': 1e-4,
    #'lambda_forecast': 0.2, 
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

class TrainingLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, 'training_log.txt')
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        with open(self.log_path, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

def save_plots(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'], label='Val')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], label='Train')
    ax2.plot(epochs, history['val_acc'], label='Val')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- PERBAIKAN METRIK MPJPE & MPJVE ---
def compute_mpjpe(pred, gt, mask,  img_w=640, img_h=480):
    """
    Mean Per Joint Position Error.
    Input shapes: [Batch, Time, 34] (Flattened features)
    """
    B, T, _ = pred.shape
    num_nodes = 17
    
    # 1. Reshape ke [B, T, 17, 2] agar bisa hitung jarak (x,y)
    pred = pred.view(B, T, num_nodes, 2)
    gt = gt.view(B, T, num_nodes, 2)
    mask = mask.view(B, T, num_nodes, 2)
    
     # ===== DENORMALIZE =====
    pred_x = (pred[..., 0] + 1) * 0.5 * img_w
    pred_y = (pred[..., 1] + 1) * 0.5 * img_h
    gt_x   = (gt[..., 0]   + 1) * 0.5 * img_w
    gt_y   = (gt[..., 1]   + 1) * 0.5 * img_h

    pred_pix = torch.stack([pred_x, pred_y], dim=-1)
    gt_pix   = torch.stack([gt_x, gt_y], dim=-1)

    node_mask = mask[..., 0]

    diff = torch.norm(pred_pix - gt_pix, p=2, dim=-1)
    diff = diff * node_mask

    return diff.sum() / (node_mask.sum() + 1e-8)

def compute_mpjve(pred, gt, mask, img_w=640, img_h=480):
    """
    Mean Per Joint Velocity Error (PIXEL).
    pred, gt: [B, T, 34] (normalized -1..1)
    """
    B, T, _ = pred.shape
    num_nodes = 17

    pred = pred.view(B, T, num_nodes, 2)
    gt   = gt.view(B, T, num_nodes, 2)
    mask = mask.view(B, T, num_nodes, 2)

    # ===== DENORMALIZE =====
    pred = torch.stack([
        (pred[..., 0] + 1) * 0.5 * img_w,
        (pred[..., 1] + 1) * 0.5 * img_h
    ], dim=-1)

    gt = torch.stack([
        (gt[..., 0] + 1) * 0.5 * img_w,
        (gt[..., 1] + 1) * 0.5 * img_h
    ], dim=-1)

    # ===== VELOCITY =====
    vel_pred = pred[:, 1:] - pred[:, :-1]
    vel_gt   = gt[:, 1:] - gt[:, :-1]

    vel_mask = mask[:, 1:, :, 0]

    diff = torch.norm(vel_pred - vel_gt, p=2, dim=-1)
    diff = diff * vel_mask

    return diff.sum() / (vel_mask.sum() + 1e-8)


def train_epoch(model, loader, crit_cls, crit_reg, optimizer, device, scaler, edge_index, use_amp):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_frames = 0
    total_mpjpe = 0.0
    total_mpjve = 0.0

    for src, mask, lbl, trg in tqdm(loader, desc="Train", leave=False):
        src  = src.to(device)
        mask = mask.to(device)
        lbl  = lbl.to(device)
        trg  = trg.to(device)

        optimizer.zero_grad()

        # =========================
        # SAMPLE-LEVEL VALID MASK
        # =========================
        valid_src = (mask.sum(dim=(1, 2)) > 0)      # (B,)
        w = valid_src.float().unsqueeze(1)          # (B,1)

        with autocast(enabled=use_amp):
            pred_cls, pred_forecast = model(src, mask, edge_index)
            # pred_cls: (B, T_pred, C)
            # lbl     : (B, T_lbl)

            # =========================
            # ALIGN TEMPORAL LENGTH
            # =========================
            T = min(pred_cls.size(1), lbl.size(1))
            pred_cls = pred_cls[:, :T]
            lbl = lbl[:, :T]

            # =========================
            # FRAME-WISE CLASSIFICATION LOSS
            # =========================
            ce_all = crit_cls(
                pred_cls.reshape(-1, pred_cls.size(-1)),
                lbl.reshape(-1)
            ).view(lbl.shape)                       # (B,T)

            loss_cls = (ce_all * w).sum() / (w.sum() * T + 1e-8)

            # =========================
            # FORECAST LOSS
            # =========================
            loss_reg = crit_reg(pred_forecast, trg)
            loss = loss_cls + loss_reg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # =========================
        # FRAME-WISE ACCURACY
        # =========================
        with torch.no_grad():
            pred_label = torch.argmax(pred_cls, dim=2)  # (B,T)
            total_correct += ((pred_label == lbl) & valid_src.unsqueeze(1)).sum().item()
            total_frames += valid_src.sum().item() * T

            total_mpjpe += compute_mpjpe(pred_forecast, trg, mask).item()
            total_mpjve += compute_mpjve(pred_forecast, trg, mask).item()

    return (
        total_loss / len(loader),
        total_correct / max(total_frames, 1),
        total_mpjpe / len(loader),
        total_mpjve / len(loader),
    )

def validate(model, loader, crit_cls, crit_reg, device, edge_index):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_frames = 0

    total_mpjpe = 0.0
    total_mpjve = 0.0

    with torch.no_grad():
        for src, mask, lbl, trg in loader:
            src  = src.to(device)
            mask = mask.to(device)
            lbl  = lbl.to(device)
            trg  = trg.to(device)

            # =========================
            # Forward
            # =========================
            pred_cls, pred_forecast = model(src, mask, edge_index)
            # pred_cls: (B, T_pred, C)
            # lbl     : (B, T_lbl)

            # =========================
            # ALIGN TEMPORAL LENGTH
            # =========================
            T = min(pred_cls.size(1), lbl.size(1))
            pred_cls = pred_cls[:, :T]
            lbl = lbl[:, :T]

            # =========================
            # SAMPLE-LEVEL VALID MASK
            # =========================
            valid_src = (mask.sum(dim=(1, 2)) > 0)      # (B,)
            w = valid_src.float().unsqueeze(1)          # (B,1)

            # =========================
            # FRAME-WISE CLASSIFICATION LOSS
            # =========================
            ce_all = crit_cls(
                pred_cls.reshape(-1, pred_cls.size(-1)),
                lbl.reshape(-1)
            ).view(lbl.shape)                           # (B,T)

            loss_cls = (ce_all * w).sum() / (w.sum() * T + 1e-8)

            # =========================
            # FORECAST LOSS
            # =========================
            loss_reg = crit_reg(pred_forecast, trg)
            loss = loss_cls +  loss_reg

            total_loss += loss.item()

            # =========================
            # FRAME-WISE ACCURACY
            # =========================
            pred_label = torch.argmax(pred_cls, dim=2)  # (B,T)
            total_correct += ((pred_label == lbl) & valid_src.unsqueeze(1)).sum().item()
            total_frames += valid_src.sum().item() * T

            # =========================
            # METRICS
            # =========================
            total_mpjpe += compute_mpjpe(pred_forecast, trg, mask).item()
            total_mpjve += compute_mpjve(pred_forecast, trg, mask).item()

    return (
        total_loss / len(loader),
        total_correct / max(total_frames, 1),
        total_mpjpe / len(loader),
        total_mpjve / len(loader)
    )


def main():
    # Setup
    use_amp = (CONFIG['device'] == 'cuda')
    base_dir = os.path.join(CONFIG['results_dir'], CONFIG['experiment_name'])
    os.makedirs(base_dir, exist_ok=True)
    logger = TrainingLogger(base_dir)
    logger.log(f"Config: {json.dumps(CONFIG, indent=2)}")
    
    # Load Data & Graph
    loaders = create_dataloaders(CONFIG['train_path'], CONFIG['val_path'], batch_size=CONFIG['batch_size'])
    edge_index = get_skeleton_edge_index().to(CONFIG['device'])
    
    # Init Model
    # Pastikan CONFIG punya 'out_channels'
    model = FallPredictionSTGCN(
        num_nodes=CONFIG['num_nodes'],
        in_channels=CONFIG['in_channels'],
        hidden_channels=CONFIG['hidden_channels'],
        kernel_size=CONFIG['kernel_size'],
        forecast_window=CONFIG['forecast_window'],
        num_classes=CONFIG['num_classes']
    ).to(CONFIG['device']) 
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scaler = GradScaler()
    
    # Loss
    class_weights = torch.tensor([1.0, 2.0, 1.5]).to(CONFIG['device']) 
    criterion_cls = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1,
    reduction="none")
    criterion_reg = nn.MSELoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    best_val_loss = float('inf')
    
    # Training Loop
    for epoch in range(1, CONFIG['epochs'] + 1):
        t_loss, t_acc, t_mpjpe, t_mpjve = train_epoch(model, loaders['train'], criterion_cls, criterion_reg, optimizer, CONFIG['device'], scaler, edge_index, use_amp)
        v_loss, v_acc, v_mpjpe, v_mpjve = validate(model, loaders['val'], criterion_cls, criterion_reg, CONFIG['device'], edge_index)

        logger.log(
            f"Ep {epoch}/{CONFIG['epochs']} | "
            f"Train: L={t_loss:.4f} A={t_acc:.4f} MPJPE={t_mpjpe:.4f} MPJVE={t_mpjve:.4f} | "
            f"Val: L={v_loss:.4f} A={v_acc:.4f} MPJPE={v_mpjpe:.4f} MPJVE={v_mpjve:.4f}"
        )

        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
        save_plots(history, os.path.join(base_dir, 'training_plot.png'))
        
        # Save Best Model Only
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
            logger.log(f"--> New Best Model Saved! Acc: {best_acc:.4f}")
        # Best by validation loss
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), os.path.join(base_dir, 'best_loss_model.pth'))


    logger.log("Training Completed.")

if __name__ == "__main__":
    main()
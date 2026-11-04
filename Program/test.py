import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ===============================
# IMPORT MODULES
# ===============================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Program_8_5_1.utils.getDataset import create_dataloaders, get_skeleton_edge_index
from Models.ST_GCN import FallPredictionSTGCN

# ===============================
# CONFIGURATION
# ===============================
CONFIG = {
    'experiment_name': 'STGCN_V2_TEST',
    'model_path': 'Coba2_pixel/STGCN_V2/best_loss_model.pth',
    'test_path': 'URFD/splits_mask/urfd_mask_test_in25_out25_st5.pkl',
    'results_dir': 'Coba2_pixel/Evaluation1',

    # HARUS SAMA DENGAN TRAINING
    'num_nodes': 17,
    'in_channels': 2,
    'hidden_channels': 16,
    'kernel_size': 3,
    'forecast_window': 25,
    'num_classes': 3,

    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'img_w': 640,
    'img_h': 480,
}

# ===============================
# SKELETON GRAPH
# ===============================
SKELETON_EDGES =[
        (0, 1), (0, 2), (1, 3), (2, 4),     # Head connections
        (5, 6),                             # Shoulder to shoulder
        (5, 11), (6, 12), (11, 12),         # Torso
        (5, 7), (7, 9),                     # Left arm
        (6, 8), (8, 10),                    # Right arm  
        (11, 13), (13, 15),                 # Left leg
        (12, 14), (14, 16),                 # Right leg
        (0, 5), (0, 6)                      # Head to torso
    ]
# ===============================
# VISUALIZATION
# ===============================
from matplotlib.lines import Line2D

CLASS_NAMES = {0: "ADL", 1: "FALLING", 2: "FALLEN"}
CLASS_COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#F44336"}  # biru, oranye, merah
def visualize_sequence(gt_seq, pred_seq, save_path,
                       frame_preds, frame_labels):
    """
    frame_preds  : array int (T,) → class index 0,1,2
    frame_labels : array int (T,) → class index 0,1,2
    """

    T_vis = min(gt_seq.shape[0], pred_seq.shape[0],
                len(frame_preds), len(frame_labels))

    gt_seq       = gt_seq[:T_vis].copy()
    pred_seq     = pred_seq[:T_vis].copy()
    frame_preds  = np.array(frame_preds[:T_vis])
    frame_labels = np.array(frame_labels[:T_vis])

    gt_seq[..., 0]   = (gt_seq[..., 0]   + 1) * 0.5 * CONFIG['img_w']
    gt_seq[..., 1]   = (gt_seq[..., 1]   + 1) * 0.5 * CONFIG['img_h']
    pred_seq[..., 0] = (pred_seq[..., 0] + 1) * 0.5 * CONFIG['img_w']
    pred_seq[..., 1] = (pred_seq[..., 1] + 1) * 0.5 * CONFIG['img_h']

    for t in range(T_vis):
        gt_c   = gt_seq[t]   - gt_seq[t].mean(axis=0)
        pred_c = pred_seq[t] - pred_seq[t].mean(axis=0)
        scale  = np.linalg.norm(gt_c) / (np.linalg.norm(pred_c) + 1e-8)
        pred_seq[t] = pred_c * scale + gt_seq[t].mean(axis=0)

    frames = sorted(set([0, T_vis//4, T_vis//2, 3*T_vis//4, T_vis - 1]))

    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
    if len(frames) == 1:
        axes = [axes]

    for idx, f in enumerate(frames):
        ax = axes[idx]

        for (i, j) in SKELETON_EDGES:
            ax.plot([gt_seq[f,i,0],   gt_seq[f,j,0]],
                    [gt_seq[f,i,1],   gt_seq[f,j,1]], color='green', linewidth=2)
            ax.plot([pred_seq[f,i,0], pred_seq[f,j,0]],
                    [pred_seq[f,i,1], pred_seq[f,j,1]], color='red',   linewidth=2)

        ax.scatter(gt_seq[f,:,0],   gt_seq[f,:,1],   c='green', s=15, zorder=3)
        ax.scatter(pred_seq[f,:,0], pred_seq[f,:,1], c='red',   s=15, zorder=3)

        pred_name = CLASS_NAMES[int(frame_preds[f])]
        gt_name   = CLASS_NAMES[int(frame_labels[f])]
        correct   = "✓" if frame_preds[f] == frame_labels[f] else "✗"

        ax.set_title(
            f"Frame {f + 1}  {correct}\n"
            f"Pred: {pred_name}\n"
            f"GT  : {gt_name}",
            fontsize=9
        )
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis("off")

    seq_pred = CLASS_NAMES[int(np.bincount(frame_preds.astype(int)).argmax())]
    seq_gt   = CLASS_NAMES[int(np.bincount(frame_labels.astype(int)).argmax())]

    fig.suptitle(
        f"Sequence Summary  |  Pred: {seq_pred}  |  GT: {seq_gt}",
        fontsize=13, y=1.04
    )

    legend_elements = [
        Line2D([0], [0], color='green', lw=2, marker='o', markersize=6, label='Ground Truth'),
        Line2D([0], [0], color='red',   lw=2, marker='o', markersize=6, label='Predicted'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.00), fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def compute_mpjpe(pred, gt, mask, img_w, img_h):
    """
    Pixel-based Mean Per Joint Position Error (masked)
    """

    B, T, _ = pred.shape
    num_nodes = 17

    pred = pred.view(B, T, num_nodes, 2)
    gt   = gt.view(B, T, num_nodes, 2)
    mask = mask.view(B, T, num_nodes, 2)

    # =========================
    # Convert to pixel space
    # =========================
    pred_x = (pred[..., 0] + 1) * 0.5 * img_w
    pred_y = (pred[..., 1] + 1) * 0.5 * img_h

    gt_x = (gt[..., 0] + 1) * 0.5 * img_w
    gt_y = (gt[..., 1] + 1) * 0.5 * img_h

    pred_pixel = torch.stack([pred_x, pred_y], dim=-1)
    gt_pixel   = torch.stack([gt_x, gt_y], dim=-1)

    node_mask = mask[..., 0]

    diff = torch.norm(pred_pixel - gt_pixel, dim=-1)
    diff = diff * node_mask

    return diff.sum() / (node_mask.sum() + 1e-8)


def compute_mpjve(pred, gt, mask, img_w, img_h):
    """
    Pixel-based Mean Per Joint Velocity Error (masked)
    """

    B, T, _ = pred.shape
    num_nodes = 17

    pred = pred.view(B, T, num_nodes, 2)
    gt   = gt.view(B, T, num_nodes, 2)
    mask = mask.view(B, T, num_nodes, 2)

    # Convert to pixel
    pred_x = (pred[..., 0] + 1) * 0.5 * img_w
    pred_y = (pred[..., 1] + 1) * 0.5 * img_h

    gt_x = (gt[..., 0] + 1) * 0.5 * img_w
    gt_y = (gt[..., 1] + 1) * 0.5 * img_h

    pred_pixel = torch.stack([pred_x, pred_y], dim=-1)
    gt_pixel   = torch.stack([gt_x, gt_y], dim=-1)

    # Velocity
    vel_pred = pred_pixel[:, 1:] - pred_pixel[:, :-1]
    vel_gt   = gt_pixel[:, 1:] - gt_pixel[:, :-1]

    vel_mask = mask[:, 1:, :, 0]

    diff = torch.norm(vel_pred - vel_gt, dim=-1)
    diff = diff * vel_mask

    return diff.sum() / (vel_mask.sum() + 1e-8)

# ===============================
# EVALUATION FUNCTION
# ===============================
def evaluate(model, loader, device, edge_index):
    model.eval()

    all_preds = []
    all_labels = []

    total_forecast_loss = 0.0
    total_cls_loss = 0.0
    total_mpjpe = 0.0
    total_mpjve = 0.0

    total_correct = 0
    total_frames = 0

    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for src, mask, lbl, trg in tqdm(loader, desc="Testing"):
            src = src.to(device)
            mask = mask.to(device)
            lbl = lbl.to(device)
            trg = trg.to(device)

            pred_cls, pred_forecast = model(src, mask, edge_index)

            # =========================
            # FORECAST LOSS
            # =========================
            total_forecast_loss += criterion_reg(pred_forecast, trg).item()

            # =========================
            # ALIGN TEMPORAL LENGTH
            # =========================
            T = min(pred_cls.size(1), lbl.size(1))
            pred_cls = pred_cls[:, :T]     # (B,T,C)
            lbl = lbl[:, :T]               # (B,T)

            # =========================
            # VALID SAMPLE MASK
            # =========================
            valid_src = (mask.sum(dim=(1, 2)) > 0)  # (B,)
            w = valid_src.float().unsqueeze(1)      # (B,1)

            # =========================
            # FRAME-WISE CE LOSS
            # =========================
            ce_all = criterion_cls(
                pred_cls.reshape(-1, pred_cls.size(-1)),
                lbl.reshape(-1)
            ).view(lbl.shape)                       # (B,T)

            loss_cls = (ce_all * w).sum() / (w.sum() * T + 1e-8)
            total_cls_loss += loss_cls.item()

            # =========================
            # FRAME-WISE ACCURACY
            # =========================
            pred_frame = torch.argmax(pred_cls, dim=2)  # (B,T)

            total_correct += ((pred_frame == lbl) & valid_src.unsqueeze(1)).sum().item()
            total_frames += valid_src.sum().item() * T

            # =========================
            # CONFUSION MATRIX DATA
            # =========================
            mask_frame = valid_src.unsqueeze(1).expand_as(lbl)
            all_preds.extend(pred_frame[mask_frame].cpu().numpy())
            all_labels.extend(lbl[mask_frame].cpu().numpy())

            # =========================
            # FORECAST METRICS
            # =========================
            total_mpjpe += compute_mpjpe(
                pred_forecast, trg, mask,
                CONFIG['img_w'], CONFIG['img_h']
            ).item()

            total_mpjve += compute_mpjve(
                pred_forecast, trg, mask,
                CONFIG['img_w'], CONFIG['img_h']
            ).item()

    frame_accuracy = total_correct / max(total_frames, 1)

    return (
        np.array(all_labels),
        np.array(all_preds),
        total_cls_loss / len(loader),
        total_forecast_loss / len(loader),
        total_mpjpe / len(loader),
        total_mpjve / len(loader),
        frame_accuracy
    )



# ===============================
# CONFUSION MATRIX
# ===============================
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()


# ===============================
# ROOT ALIGNMENT (HARUS DI SINI)
# ===============================
def align_by_root(pred_seq, gt_seq, mask_seq, root_idx=11):
    """
    Align predicted skeleton to GT using root joint (hip/pelvis)
    pred_seq, gt_seq, mask_seq: (T, 17, 2)
    """
    aligned_pred = pred_seq.copy()

    for t in range(pred_seq.shape[0]):
        if mask_seq[t, root_idx, 0] > 0:
            offset = gt_seq[t, root_idx] - pred_seq[t, root_idx]
            aligned_pred[t] += offset

    return aligned_pred


# ===============================
# MAIN
# ===============================
def main():
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    loaders = create_dataloaders(
        train_path=None,
        val_path=None,
        test_path=CONFIG['test_path'],
        batch_size=CONFIG['batch_size']
    )

    test_loader = loaders['test']
    edge_index  = get_skeleton_edge_index().to(CONFIG['device'])

    model = FallPredictionSTGCN(
        num_nodes=CONFIG['num_nodes'],
        in_channels=CONFIG['in_channels'],
        hidden_channels=CONFIG['hidden_channels'],
        kernel_size=CONFIG['kernel_size'],
        forecast_window=CONFIG['forecast_window'],
        num_classes=CONFIG['num_classes']
    ).to(CONFIG['device'])

    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    print("Model loaded.")

    # ===============================
    # EVALUATION
    # ===============================
    y_true, y_pred, class_loss, forecast_loss, mpjpe, mpjve, frame_accuracy = evaluate(
        model, test_loader, CONFIG['device'], edge_index
    )

    class_names = ['ADL', 'FALLING', 'FALLEN']

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )

    print("\n=== EVALUATION RESULTS ===")
    print(f"Classification Loss (CE): {class_loss:.6f}")
    print(f"Forecast MSE           : {forecast_loss:.6f}")
    print(f"MPJPE                  : {mpjpe:.4f}")
    print(f"MPJVE                  : {mpjve:.4f}")
    print(f"Frame-wise Accuracy    : {frame_accuracy:.4f}")
    print(report)

    plot_confusion_matrix(
        y_true, y_pred,
        class_names,
        os.path.join(CONFIG['results_dir'], "confusion_matrix.png")
    )

    # ===============================
    # SEQUENCE VISUALIZATION
    # ===============================
    viz_dir = os.path.join(CONFIG['results_dir'], "skeleton_viz")
    os.makedirs(viz_dir, exist_ok=True)

    T          = CONFIG['forecast_window']
    sample_idx = 0

    model.eval()

    with torch.no_grad():
        for src, mask, lbl, trg in test_loader:
            src  = src.to(CONFIG['device'])
            mask = mask.to(CONFIG['device'])
            trg  = trg.to(CONFIG['device'])
            lbl  = lbl.to(CONFIG['device'])

            pred_cls, pred_forecast = model(src, mask, edge_index)
            pred_frame = torch.argmax(pred_cls, dim=2)          # (B, T)

            for b in range(src.size(0)):
                gt_seq   = trg[b].view(T, 17, 2).cpu().numpy()
                pred_seq = pred_forecast[b].view(T, 17, 2).cpu().numpy()
                mask_seq = mask[b].view(T, 17, 2).cpu().numpy()

                pred_seq = align_by_root(pred_seq, gt_seq, mask_seq, root_idx=11)

                visualize_sequence(
                    gt_seq      = gt_seq,
                    pred_seq    = pred_seq,
                    save_path   = os.path.join(viz_dir, f"sequence_{sample_idx:04d}.png"),
                    frame_preds = pred_frame[b].cpu().numpy(),
                    frame_labels= lbl[b].cpu().numpy(),
                )

                sample_idx += 1

    print(f"Saved {sample_idx} sequence visualizations.")

    # ===============================
    # SAVE METRICS
    # ===============================
    metrics_path = os.path.join(CONFIG['results_dir'], "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== EVALUATION RESULTS ===\n")
        f.write(f"Classification Loss (CE): {class_loss:.6f}\n")
        f.write(f"Forecast MSE           : {forecast_loss:.6f}\n")
        f.write(f"MPJPE                  : {mpjpe:.4f}\n")
        f.write(f"MPJVE                  : {mpjve:.4f}\n")
        f.write(f"Frame-wise Accuracy    : {frame_accuracy:.4f}\n\n")
        f.write("=== CLASSIFICATION REPORT ===\n")
        f.write(report)

    print(f"[OK] Metrics saved to: {metrics_path}")
    print("Testing completed.")

if __name__ == "__main__":
    main()
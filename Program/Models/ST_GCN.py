# # #1 Layer
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch_geometric_temporal.nn.attention import STConv


# # class FallPredictionSTGCN(nn.Module):
# #     def __init__(
# #         self,
# #         num_nodes,
# #         in_channels,
# #         hidden_channels,
# #         kernel_size,
# #         forecast_window,
# #         num_classes
# #     ):
# #         super().__init__()

# #         self.num_nodes = num_nodes
# #         self.forecast_window = forecast_window
# #         self.hidden_channels = hidden_channels

# #         # --- Backbone (1 STConv) ---
# #         self.st_conv = STConv(
# #             num_nodes,
# #             in_channels,
# #             hidden_channels,
# #             hidden_channels,
# #             kernel_size,
# #             K=3
# #         )

# #         # --- Classification Head ---
# #         self.cls_head = nn.Sequential(
# #             nn.Linear(hidden_channels, 64),
# #             nn.ReLU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(64, num_classes)
# #         )

# #         # --- Forecasting Head ---
# #         self.forecast_head = nn.Sequential(
# #             nn.Linear(hidden_channels, 32),
# #             nn.ReLU(),
# #             nn.Dropout(0.2),
# #             nn.Linear(32, forecast_window * num_nodes * 2)
# #         )

# #     def forward(self, x, mask, edge_index):
# #         """
# #         x    : (B, 25, 34)
# #         mask : (B, 25, 34)
# #         """
# #         B = x.size(0)

# #         # (B, T, V, C)
# #         x = x.view(B, 25, self.num_nodes, 2)
# #         mask = mask.view(B, 25, self.num_nodes, 2)

# #         # Apply mask
# #         x = x * mask

# #         # ST-GCN
# #         out = self.st_conv(x, edge_index)
# #         out = F.relu(out)

# #         # Pool over nodes -> (B, T, C)
# #         out = out.mean(dim=2)

# #         # Pool over time -> (B, C)
# #         out = out.mean(dim=1)

# #         # Heads
# #         logits = self.cls_head(out)

# #         forecast = self.forecast_head(out)
# #         forecast = forecast.view(
# #             B,
# #             self.forecast_window,
# #             self.num_nodes * 2
# #         )

# #         return logits, forecast

# #2lyer
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric_temporal.nn.attention import STConv


# class FallPredictionSTGCN(nn.Module):
#     def __init__(
#         self,
#         num_nodes,
#         in_channels,
#         hidden_channels,
#         kernel_size,
#         forecast_window,
#         num_classes
#     ):
#         super().__init__()

#         self.num_nodes = num_nodes
#         self.forecast_window = forecast_window
#         self.hidden_channels = hidden_channels

#         # ============================
#         # Backbone: 2 ST-GCN Layers
#         # ============================
#         self.st_conv1 = STConv(
#             num_nodes,
#             in_channels,
#             hidden_channels,
#             hidden_channels,
#             kernel_size,
#             K=3
#         )

#         self.st_conv2 = STConv(
#             num_nodes,
#             hidden_channels,
#             hidden_channels,
#             hidden_channels,
#             kernel_size,
#             K=3
#         )

#         # ============================
#         # Classification Head
#         # ============================
#         self.cls_head = nn.Sequential(
#             nn.Linear(hidden_channels, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )

#         # ============================
#         # Forecasting Head
#         # ============================
#         self.forecast_head = nn.Sequential(
#             nn.Linear(hidden_channels, 32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, forecast_window * num_nodes * 2)
#         )

#     def forward(self, x, mask, edge_index):
#         """
#         x    : (B, 25, 34)
#         mask : (B, 25, 34)
#         """
#         B = x.size(0)

#         # (B, T, V, C)
#         x = x.view(B, 25, self.num_nodes, 2)
#         mask = mask.view(B, 25, self.num_nodes, 2)

#         # Masking missing joints
#         x = x * mask

#         # ============================
#         # ST-GCN Backbone
#         # ============================
#         x = self.st_conv1(x, edge_index)
#         x = F.relu(x)

#         x = self.st_conv2(x, edge_index)
#         x = F.relu(x)

#         # ============================
#         # Global Pooling
#         # ============================
#         x = x.mean(dim=2)   # Pool node → (B, T, C)
#         x = x.mean(dim=1)   # Pool time → (B, C)

#         # ============================
#         # Heads
#         # ============================
#         logits = self.cls_head(x)

#         forecast = self.forecast_head(x)
#         forecast = forecast.view(
#             B,
#             self.forecast_window,
#             self.num_nodes * 2
#         )

#         return logits, forecast

#1 Layer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv


class FallPredictionSTGCN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_channels,
        hidden_channels,
        kernel_size,
        forecast_window,
        num_classes
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.forecast_window = forecast_window
        self.hidden_channels = hidden_channels

        # --- Backbone (1 STConv) ---
        self.st_conv = STConv(
            num_nodes,
            in_channels,
            hidden_channels,
            hidden_channels,
            kernel_size,
            K=3
        )

        # --- Classification Head ---
        
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # --- Forecasting Head ---
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, forecast_window * num_nodes * 2)
        )

    def forward(self, x, mask, edge_index):
        """
        x    : (B, 25, 34)
        mask : (B, 25, 34)
        """
        B = x.size(0)

        # (B, T, V, C)
        x = x.view(B, 25, self.num_nodes, 2)
        mask = mask.view(B, 25, self.num_nodes, 2)

        # Apply mask
        x = x * mask

        # ST-GCN
        out = self.st_conv(x, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)

        # Pool over nodes -> (B, T, C)
        out = out.mean(dim=2)

        # Heads
        logits = self.cls_head(out)

        pooled = out.mean(dim=1)  

        forecast = self.forecast_head(pooled)
        forecast = forecast.view(
            B,
            self.forecast_window,
            self.num_nodes * 2
        )

        return logits, forecast

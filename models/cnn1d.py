"""
1D-CNN: mô hình dự phòng nếu MLP không đủ tốt (Cổng 1, cuối tuần 2).

    (batch, 1500) -> thêm 1 chiều kênh -> (batch, 1, 1500)
    Conv1d(1 -> 16, kernel 7) -> ReLU -> MaxPool(2)      -> (batch, 16, 750)
    Conv1d(16 -> 16, kernel 5) -> ReLU -> MaxPool(2)     -> (batch, 16, 375)
    Flatten                                              -> (batch, 6000)
    Dropout -> head (Linear 6000 -> n_classes)

Khác với MLP, một bộ lọc (kernel) trượt dọc theo payload nên nhận ra cùng một mẫu byte dù nó nằm ở
vị trí nào ("GET /", "USER ", "HTTP/1.1"...). Pooling giữ giá trị lớn nhất trong từng cửa sổ 2 phần tử,
giúp chuỗi ngắn đi một nửa và bớt nhạy với dịch chuyển nhỏ.

Chạy:  python skeleton.py --model cnn1d --smoke
       python skeleton.py --model cnn1d --who <ten>
"""
import torch.nn as nn

from models.registry import N_BYTES, register


@register("cnn1d")
class CNN1D(nn.Module):
    def __init__(self, n_classes: int, channels=(16, 16), dropout: float = 0.2):
        super().__init__()
        c1, c2 = channels
        self.body = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        n_flat = c2 * (N_BYTES // 4)          # hai lần pooling(2): 1500 -> 750 -> 375;  16 * 375 = 6000
        self.head = nn.Linear(n_flat, n_classes)

    def features(self, x):
        """Vector đặc trưng ngay trước head, kích thước (batch, 6000)."""
        return self.body(x.unsqueeze(1))      # Conv1d cần (batch, kênh, độ dài)

    def forward(self, x):
        """x: (batch, 1500) số thực 0..1  ->  logits (batch, n_classes)."""
        return self.head(self.features(x))

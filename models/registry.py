"""
Sổ đăng ký mô hình dùng chung (skeleton.py import MODELS và N_BYTES từ file này).

Cách thêm một mô hình mới: tạo file models/<ten>.py rồi viết

    from models.registry import register, N_BYTES

    @register("ten_mo_hinh")
    class TenMoHinh(nn.Module):
        def __init__(self, n_classes): ...      # bắt buộc có self.head = nn.Linear(..., n_classes)

Không cần sửa file này hay skeleton.py: models/__init__.py tự import mọi file trong thư mục models/.
"""

N_BYTES = 1500     # số byte payload của mỗi gói tin = số chiều đầu vào của mọi mô hình
MODELS = {}        # tên mô hình -> lớp mô hình; skeleton.build_model(name, n_classes) tra bảng này


def register(name: str):
    """Decorator: ghi lớp mô hình vào MODELS dưới tên `name` (tên này dùng cho `--model`)."""
    def decorator(cls):
        old = MODELS.get(name)
        if old is not None and (old.__module__, old.__qualname__) != (cls.__module__, cls.__qualname__):
            raise ValueError(f"tên mô hình '{name}' đã được đăng ký bởi {old.__module__}.{old.__qualname__}")
        MODELS[name] = cls
        return cls
    return decorator

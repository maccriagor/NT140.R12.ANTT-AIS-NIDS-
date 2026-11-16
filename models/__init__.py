"""
Gói models: tự import mọi file .py trong thư mục này để từng mô hình tự đăng ký qua @register.
Nhờ vậy thêm mô hình mới chỉ cần thêm một file, không phải sửa skeleton.py hay file này.
"""
import importlib
import pkgutil

from .registry import MODELS, N_BYTES, register

for _mod in pkgutil.iter_modules(__path__):
    if _mod.name != "registry" and not _mod.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_mod.name}")

__all__ = ["MODELS", "N_BYTES", "register"]

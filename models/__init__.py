"""Import every module in models/ so each one runs its @register(...) decorator."""
import importlib
import pkgutil

for _m in pkgutil.iter_modules(__path__):
    if _m.name != "registry":
        importlib.import_module(f"{__name__}.{_m.name}")

import datetime
import torch
from modules import scripts


class BaseGuidanceScript(scripts.Script):
    """
    APG, TCFG, MaHiRoの共通機能を管理する基底クラス。
    ロギング、デバッグフラグの管理などを一元化する。
    """
    _script_name = "BaseGuidance"
    is_enabled = False
    _enable_debug_logging_ui = False
    force_all_steps_debug_log = False
    force_all_steps_debug_log_if_global_debug_on = False

    def log_message(self, message, level="INFO", step_info=None, sigma_info=None):
        """共通ログ出力メソッド。"""
        if not self.is_enabled:
            return

        is_debug_message = level == "DEBUG"
        if is_debug_message and not self._enable_debug_logging_ui:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        step_str = f"S:{step_info}" if step_info is not None else ""
        sigma_val_str = ""
        if isinstance(sigma_info, torch.Tensor) and sigma_info.numel() > 0:
            try:
                sigma_val_str = f"σ:{sigma_info.cpu().item():.4f}" if sigma_info.numel() == 1 else f"σ_shape:{sigma_info.shape}"
            except Exception:
                sigma_val_str = f"σ_shape:{sigma_info.shape}(cpu_item_error)"
        elif isinstance(sigma_info, float):
            sigma_val_str = f"σ:{sigma_info:.4f}"

        prefix_info = f"({step_str} {sigma_val_str})".strip().replace("  ", " ")
        if prefix_info == "()":
            prefix_info = ""

        print(f"{timestamp} {level} [{self._script_name}]{prefix_info} {message}")

    def should_log_debug(self, condition_is_true_for_extra_debug=True):
        """デバッグログを出力すべきかどうかを判定する。"""
        if not self.is_enabled or not self._enable_debug_logging_ui:
            return False
        if condition_is_true_for_extra_debug:
            return True
        return self.force_all_steps_debug_log_if_global_debug_on

    def title(self):
        return self._script_name

    def show(self, is_img2img):
        # scripts.AlwaysHidden は存在しないため、Falseを返すように修正
        return False

    def ui(self, is_img2img):
        return []

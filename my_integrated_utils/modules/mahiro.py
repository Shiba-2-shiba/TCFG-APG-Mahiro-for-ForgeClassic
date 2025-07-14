import gradio as gr
import torch
import torch.nn.functional as F

from modules import scripts, shared, script_callbacks
import datetime

class ScriptMahiro(scripts.ScriptBuiltinUI): # ScriptBuiltinUIのまま
    section = "cfg" # これはUI表示に影響しない
    create_group = False
    sorting_priority = 1

    _instance = None # シングルトンインスタンス管理
    
    group = gr.Group(visible=False) # ダミーのGradioグループ要素。UIには表示されないが、内部参照用
    
    def __init__(self):
        super().__init__()
        if ScriptMahiro._instance is None:
            ScriptMahiro._instance = self
        self._script_name = "MaHiRo"
        self._enable_debug_logging_ui = False # デバッグログ用のフラグ
        self.mahiro_enabled = False # UIからの有効化フラグを保持
        self.log_message("MaHiRo Script Initialized.", "INFO")

        # infotext_fieldsは統合スクリプトで管理されるため、ここでは空にする
        self.infotext_fields = []

    def log_message(self, message, level="INFO", step_info=None, sigma_info=None):
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
        if prefix_info == "()": prefix_info = ""
        
        print(f"{timestamp} {level} [{self._script_name}]{prefix_info} {message}")

    def should_log_debug(self, condition_is_true_for_extra_debug=True):
        # MaHiRoにはforce_all_steps_debug_logがないので、簡易版
        return self._enable_debug_logging_ui and condition_is_true_for_extra_debug

    def title(self):
        return "MaHiRo" # UI上は単独では表示されないため、これは内部的な名前

    def show(self, is_img2img):
        # 修正: scripts.AlwaysHidden の代わりに整数値 1 を返す
        return 1 # 統合スクリプトがUIを制御するため、MaHiRo自身は常に非表示

    def ui(self, is_img2img):
        # 統合スクリプトがUIを制御するため、MaHiRo自身のUIは空
        return []

    def process(self, p, enable_mahiro, enable_debug_logging_ui):
        # 統合スクリプトから有効化フラグとデバッグログフラグを受け取る
        self.mahiro_enabled = enable_mahiro
        self._enable_debug_logging_ui = enable_debug_logging_ui
        self.log_message(f"MaHiRo process() called. Enabled: {self.mahiro_enabled}, Debug Logging: {self._enable_debug_logging_ui}", "INFO")

        # infotextは統合スクリプトで処理されるため、ここでは不要だが、もし独自に持ちたいなら
        # p.extra_generation_params.update({"MaHiRo": self.mahiro_enabled})
        pass # MaHiRoはprocess_before_every_samplingでフックせず、統合関数から直接呼ばれる

    def process_before_every_sampling(self, p, enable, *args, **kwargs):
        # このメソッドは統合スクリプトによって制御されるため、何もしないか、削除する
        # ここでは、MaHiRoが独自にhookしないようにする
        if self.should_log_debug():
            self.log_message("MaHiRo's process_before_every_sampling called, but it should be handled by integration.", "DEBUG")
        pass


    @torch.inference_mode()
    # この関数は`_integrated_cfg_function`から直接呼び出されるため、インスタンスメソッドとして定義する
    def mahiro_normd(self, args: dict):
        step = getattr(shared.state, 'sampling_step', -1)
        sigma = args.get("sigma", torch.tensor([-1.0]))
        if isinstance(sigma, torch.Tensor) and sigma.numel() == 1:
            sigma_val = sigma.item()
        else:
            sigma_val = -1.0 # Or handle batch sigma

        if self.mahiro_enabled: # 統合スクリプトからのフラグを参照
            if self.should_log_debug(True):
                self.log_message(f"mahiro_normd invoked.", "DEBUG", step, sigma_val)
                # デバッグログの出力 (APG/TCFGに合わせてlog_tensor_info_via_instanceを実装するか、直接出力)
                # 仮に、APGForgeから提供されるlog_tensor_info_via_instanceを使う場合は、その関数をインポートするか、
                # ここで単純にprintする。ここでは簡易的にprint
                self.log_message(f"[MaHiRo] cond_scale: {args['cond_scale']}", "DEBUG", step, sigma_val)
                self.log_message(f"[MaHiRo] cond_denoised shape: {args['cond_denoised'].shape}, norm: {args['cond_denoised'].norm().item():.4f}", "DEBUG", step, sigma_val)
                self.log_message(f"[MaHiRo] uncond_denoised shape: {args['uncond_denoised'].shape}, norm: {args['uncond_denoised'].norm().item():.4f}", "DEBUG", step, sigma_val)
                self.log_message(f"[MaHiRo] denoised shape: {args['denoised'].shape}, norm: {args['denoised'].norm().item():.4f}", "DEBUG", step, sigma_val)

            scale: float = args["cond_scale"]
            cond_p: torch.Tensor = args["cond_denoised"]
            uncond_p: torch.Tensor = args["uncond_denoised"]
            leap = cond_p * scale
            u_leap = uncond_p * scale
            cfg: torch.Tensor = args["denoised"] # TCFG+APG後のdenoised結果
            merge = (leap + cfg) / 2
            normu = torch.sqrt(u_leap.abs()) * u_leap.sign()
            normm = torch.sqrt(merge.abs()) * merge.sign()
            sim = F.cosine_similarity(normu, normm).mean()
            simsc = 2 * (sim + 1)
            wm = (simsc * cfg + (4 - simsc) * leap) / 4
            
            if self.should_log_debug(True):
                self.log_message(f"[MaHiRo] wm norm: {wm.norm().item():.4f}", "DEBUG", step, sigma_val)

            return wm
        else:
            # MaHiRoが無効な場合、何もしないでdenoised結果をそのまま返す
            if self.should_log_debug(True):
                self.log_message(f"MaHiRo disabled, returning original denoised.", "DEBUG", step, sigma_val)
            return args["denoised"]

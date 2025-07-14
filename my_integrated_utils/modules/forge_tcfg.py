import torch
import gradio as gr
from modules import scripts, shared, script_callbacks # Forgeの標準モジュール
import datetime # ログのタイムスタンプ用
import traceback # エラー発生時の詳細情報取得用
import os # 基本的なOS機能

# TCFG: Tangential Damping Classifier-free Guidance - (arXiv: https://arxiv.org/abs/2503.18137)

class TCFGForge(scripts.Script):
    _instance = None # シングルトンインスタンス管理
    
    group = gr.Group(visible=False) # ダミーのGradioグループ要素。UIには表示されないが、内部参照用

    def __init__(self):
        super().__init__()
        if TCFGForge._instance is None:
            TCFGForge._instance = self

        self.tcfg_enabled = False
        self._enable_debug_logging_ui = False
        self.force_all_steps_debug_log = False
        self._script_name = "TCFGForge"
        self.force_all_steps_debug_log_if_global_debug_on = False
        self.user_cfg_scale_for_first_step_override = None

        print(f"[{self._script_name}] Initializing script instance...")
        script_callbacks.on_model_loaded(self.on_model_loaded_instance_method)
        script_callbacks.on_script_unloaded(self.on_script_unloaded_instance_method)
        self.log_message("TCFGForge Script Initialized.", "INFO")

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
        return self._enable_debug_logging_ui and \
               (condition_is_true_for_extra_debug or self.force_all_steps_debug_log_if_global_debug_on)

    def title(self):
        return "Tangential Damping CFG (TCFG) - Forge"

    def show(self, is_img2img):
        # 修正: scripts.AlwaysHidden の代わりに整数値 1 を返す
        return 1 # 統合スクリプトでUIを制御するためHiddenにする

    def ui(self, is_img2img):
        return [] # 統合スクリプトでUIを制御するため空にする

    def elem_id_prefix(self, is_img2img):
        return f"tcfg_forge_script_{'img2img' if is_img2img else 'txt2img'}"

    def on_model_loaded_instance_method(self, model):
        self._reset_tcfg_state()
        model_name = "Unknown Model"
        if hasattr(model, 'sd_model_checkpoint') and model.sd_model_checkpoint:
            model_name = os.path.basename(model.sd_model_checkpoint)
        elif hasattr(shared, 'sd_model') and hasattr(shared.sd_model, 'sd_model_checkpoint') and shared.sd_model.sd_model_checkpoint:
             model_name = os.path.basename(shared.sd_model.sd_model_checkpoint)
        self.log_message(f"TCFG state reset due to model load/change: {model_name}", "INFO")

    def on_script_unloaded_instance_method(self):
        self.log_message("Script unloaded. Removing TCFGForge._instance.", "INFO")
        if TCFGForge._instance == self:
            TCFGForge._instance = None

    def _reset_tcfg_state(self):
        if self._enable_debug_logging_ui:
            self.log_message(f"TCFG internal state reset.", "DEBUG")

    def process(self, p,
                enable_debug_logging_ui,
                force_all_steps_debug_log_ui,
                tcfg_enabled):
        
        self._enable_debug_logging_ui = enable_debug_logging_ui
        self.force_all_steps_debug_log = force_all_steps_debug_log_ui
        self.force_all_steps_debug_log_if_global_debug_on = self._enable_debug_logging_ui and self.force_all_steps_debug_log
        
        self.log_message(f"TCFG process() called. Debug Logging UI: {self._enable_debug_logging_ui}, Force All Steps Log Active: {self.force_all_steps_debug_log_if_global_debug_on}", "INFO")

        self.tcfg_enabled = tcfg_enabled
        if self.should_log_debug(True):
            self.log_message(f"TCFG params captured: enabled={self.tcfg_enabled}", "DEBUG")

        # infotextは統合スクリプトで管理するため、ここでは設定しない
        # p.extra_generation_params["TCFG Debug Logging"] = self._enable_debug_logging_ui
        # ...

    def process_before_every_sampling(self, p, *args, **kwargs):
        # 修正: このメソッドは統合スクリプトによって制御されるため、
        # モデルパッチングのロジックを完全に削除する
        if self.should_log_debug():
            self.log_message(f"TCFG's process_before_every_sampling called, but it should be handled by integration.", "DEBUG")

        current_sampling_step = getattr(shared.state, 'sampling_step', 0)
        is_first_step_of_a_pass = (current_sampling_step == 0)

        if is_first_step_of_a_pass:
            self._reset_tcfg_state()
            if hasattr(p, 'cfg_scale'):
                self.user_cfg_scale_for_first_step_override = p.cfg_scale
                if self.should_log_debug(True):
                    self.log_message(f"Captured user CFG scale for pass: {self.user_cfg_scale_for_first_step_override}", "DEBUG")
            else:
                self.user_cfg_scale_for_first_step_override = None
                if self.should_log_debug(True):
                    self.log_message(f"p.cfg_scale not found. Cannot determine user CFG scale for override.", "WARNING")
        
        # モデルパッチングのロジックは統合スクリプトに移管されたため、ここから削除
        # model_patcher.set_model_sampler_cfg_function(self.tcfg_cfg_function) # 統合スクリプトが呼ぶ


    def log_tensor_info_via_instance(self, name, tensor, step, sigma, force_log_early_step=True):
        should_log_this_detail = self.force_all_steps_debug_log_if_global_debug_on or \
                                 (self._enable_debug_logging_ui and step < 2 and force_log_early_step)
        
        if not should_log_this_detail and not (self._enable_debug_logging_ui and self.force_all_steps_debug_log):
             if not (self._enable_debug_logging_ui and step < 2 and force_log_early_step and self.should_log_debug()):
                if not self.should_log_debug(False):
                     return


        log_msg_prefix = f"[{name}]"
        if isinstance(tensor, torch.Tensor):
            norm_val = "N/A (empty)"
            has_nan_val = "N/A (empty)"
            if tensor.numel() > 0 :
                try:
                    norm_val = f"{tensor.norm().item():.4f}"
                    has_nan_val = str(torch.isnan(tensor).any().item())
                except Exception as e:
                    norm_val = f"Error: {e}"
                    has_nan_val = f"Error: {e}"
            self.log_message(f"{log_msg_prefix}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, norm={norm_val}, has_nan={has_nan_val}", "DEBUG", step, sigma)
        elif isinstance(tensor, (int, float, bool)):
            self.log_message(f"{log_msg_prefix}: value={tensor}", "DEBUG", step, sigma)
        else:
            self.log_message(f"{log_msg_prefix}: type={type(tensor)}", "DEBUG", step, sigma)


    def score_tangential_damping(self, cond_score: torch.Tensor, uncond_score: torch.Tensor, step: int = -1, sigma: float = -1.0) -> torch.Tensor:
        batch_num = cond_score.shape[0]
        cond_score_flat = cond_score.reshape(batch_num, 1, -1).float()
        uncond_score_flat = uncond_score.reshape(batch_num, 1, -1).float()

        score_matrix = torch.cat((uncond_score_flat, cond_score_flat), dim=1)
        
        if self.should_log_debug():
            self.log_message(f"[score_tangential_damping]: cond_score_flat shape: {cond_score_flat.shape}, uncond_score_flat shape: {uncond_score_flat.shape}", "DEBUG", step, sigma)
            self.log_message(f"[score_tangential_damping]: score_matrix shape: {score_matrix.shape}", "DEBUG", step, sigma)

        try:
            _, _, Vh = torch.linalg.svd(score_matrix, full_matrices=False)
            if self.should_log_debug():
                self.log_message(f"[score_tangential_damping]: SVD performed on device: {score_matrix.device}", "DEBUG", step, sigma)
        except RuntimeError as e:
            self.log_message(f"[score_tangential_damping]: RuntimeError during SVD, falling back to CPU: {e}", "WARNING", step, sigma)
            _, _, Vh = torch.linalg.svd(score_matrix.cpu(), full_matrices=False)
            Vh = Vh.to(uncond_score_flat.device)
            if self.should_log_debug():
                self.log_message(f"[score_tangential_damping]: SVD performed on CPU, result moved to {uncond_score_flat.device}", "DEBUG", step, sigma)


        v1 = Vh[:, 0:1, :].to(uncond_score_flat.device)
        
        if self.should_log_debug():
            self.log_message(f"[score_tangential_damping]: Vh shape: {Vh.shape}, v1 shape: {v1.shape}", "DEBUG", step, sigma)


        uncond_score_td = (uncond_score_flat @ v1.transpose(-2, -1)) * v1
        
        if self.should_log_debug():
            self.log_message(f"[score_tangential_damping]: uncond_score_td shape before reshape: {uncond_score_td.shape}", "DEBUG", step, sigma)


        return uncond_score_td.reshape_as(uncond_score).to(uncond_score.dtype)


    @torch.inference_mode() # <-- ここにデコレータを追加 (元からあるが念のため)
    def tcfg_cfg_function(self, args):
        # print("--- TCFG_CFG_FUNCTION ENTERED ---") # 無条件プリントはログ機構に任せる


        step = getattr(shared.state, 'sampling_step', -1)
        sigma_tensor_arg = args.get("sigma", torch.tensor([-1.0]))
        if not isinstance(sigma_tensor_arg, torch.Tensor):
            sigma_tensor = torch.tensor([sigma_tensor_arg], device=args["input"].device, dtype=args["input"].dtype)
        else:
            sigma_tensor = sigma_tensor_arg.to(args["input"].device, dtype=args["input"].dtype)
        sigma_val = sigma_tensor[0].item() if sigma_tensor.numel() == 1 else -1.0


        should_log_details_for_this_step = self.should_log_debug(step < 2 or self.force_all_steps_debug_log_if_global_debug_on)


        if should_log_details_for_this_step:
            self.log_message(f"tcfg_cfg_function invoked.", "DEBUG", step, sigma_val)
            self.log_tensor_info_via_instance("args['cond_denoised'] (x0_cond)", args["cond_denoised"], step, sigma_val, force_log_early_step=False)
            self.log_tensor_info_via_instance("args['uncond_denoised'] (x0_uncond)", args["uncond_denoised"], step, sigma_val, force_log_early_step=False)
            self.log_tensor_info_via_instance("args['input'] (x_t)", args["input"], step, sigma_val, force_log_early_step=False)
            self.log_message(f"  Cond Scale (w_sampler): {args['cond_scale']:.2f}", "DEBUG", step, sigma_val)


        cond_pred = args["cond_denoised"]
        uncond_pred = args["uncond_denoised"]
        x_input = args["input"]


        cond_score_noise_pred = x_input - cond_pred
        uncond_score_noise_pred = x_input - uncond_pred


        if should_log_details_for_this_step:
            self.log_tensor_info_via_instance("Original Cond Noise Score", cond_score_noise_pred, step, sigma_val, force_log_early_step=False)
            self.log_tensor_info_via_instance("Original Uncond Noise Score", uncond_score_noise_pred, step, sigma_val, force_log_early_step=False)


        uncond_td_noise_score = self.score_tangential_damping(
            cond_score_noise_pred,
            uncond_score_noise_pred,
            step=step,
            sigma=sigma_val
        )


        if should_log_details_for_this_step:
            self.log_tensor_info_via_instance("Uncond Noise Score (Tangential Damped)", uncond_td_noise_score, step, sigma_val, force_log_early_step=False)


        effective_cond_scale = args["cond_scale"]
        is_true_first_step_of_pass = (step == 0) # TCFGはprev_sigmaを見てないので、これでOK
        if is_true_first_step_of_pass and \
           effective_cond_scale == 1.0 and \
           self.user_cfg_scale_for_first_step_override is not None and \
           self.user_cfg_scale_for_first_step_override != 1.0:
            
            effective_cond_scale = self.user_cfg_scale_for_first_step_override
            if self.should_log_debug(True):
                self.log_message(f"FIRST STEP CFG OVERRIDE: Sampler CFG was {args['cond_scale']:.2f}, using stored user CFG {effective_cond_scale:.2f}.", "DEBUG", step, sigma_val)


        final_noise_prediction = uncond_td_noise_score + effective_cond_scale * (cond_score_noise_pred - uncond_td_noise_score)


        if should_log_details_for_this_step:
            self.log_tensor_info_via_instance("Final Noise Prediction with TCFG", final_noise_prediction, step, sigma_val, force_log_early_step=False)


        return final_noise_prediction

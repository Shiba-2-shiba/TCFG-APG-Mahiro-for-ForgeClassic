import torch
import gradio as gr
from modules import scripts, shared, script_callbacks # Forgeの標準モジュール
import datetime # ログのタイムスタンプ用
import traceback # エラー発生時の詳細情報取得用
import os # 基本的なOS機能

# --- APG Core Logic ---
def project_apg(v0, v1, script_instance=None, step=-1, sigma=-1.0):
    """
    Projects vector v0 onto vector v1.
    v0: The vector to be projected (e.g., guidance).
    v1: The vector onto which v0 is projected (e.g., cond_denoised).
    Returns:
        v0_parallel: The component of v0 parallel to v1.
        v0_orthogonal: The component of v0 orthogonal to v1.
    """
    dims_to_operate = [d for d in range(1, v0.ndim)]
    if not dims_to_operate:
        if script_instance and script_instance.should_log_debug():
            script_instance.log_message(f"[project_apg]: Warning - no dimensions to operate on. v0.ndim: {v0.ndim}", "DEBUG", step, sigma)
        dims_to_operate = [0] # Fallback for 1D tensors, though typically not expected for latents

    # Ensure v0 and v1 are tensors before calling .norm() or other tensor ops
    if not isinstance(v0, torch.Tensor) or not isinstance(v1, torch.Tensor):
        if script_instance and script_instance.should_log_debug():
            script_instance.log_message(f"[project_apg]: Error - v0 or v1 is not a tensor. v0 type: {type(v0)}, v1 type: {type(v1)}", "ERROR", step, sigma)
        return torch.zeros_like(v0) if isinstance(v0, torch.Tensor) else 0, v0

    v1_norm = v1.norm(p=2, dim=dims_to_operate, keepdim=True)
    if (v1_norm < 1e-8).any():
        if script_instance and script_instance.should_log_debug():
            script_instance.log_message(f"[project_apg]: Warning - v1 norm ({v1_norm.mean().item():.4f}) is close to zero for at least one item, cannot normalize. Returning v0 as orthogonal.", "DEBUG", step, sigma)
        return torch.zeros_like(v0), v0

    v1_normalized = v1 / (v1_norm + 1e-8)
    
    try:
        v0_parallel = (v0 * v1_normalized).sum(dim=dims_to_operate, keepdim=True) * v1_normalized
    except RuntimeError as e:
        if script_instance and script_instance.should_log_debug():
            script_instance.log_message(f"[project_apg]: RuntimeError during parallel component calculation: {e}. v0 shape: {v0.shape}, v1_normalized shape: {v1_normalized.shape}", "ERROR", step, sigma)
        return torch.zeros_like(v0), v0 # Fallback

    v0_orthogonal = v0 - v0_parallel

    log_condition_met = (step < 2 and script_instance.force_all_steps_debug_log_if_global_debug_on) or \
                        (script_instance.force_all_steps_debug_log and script_instance._enable_debug_logging_ui)

    if script_instance and script_instance.should_log_debug(log_condition_met or (step < 2)):
        script_instance.log_message(f"[project_apg]: v0 norm: {v0.norm().item():.4f}, v1 input norm: {v1.norm().item():.4f}", "DEBUG", step, sigma)
        script_instance.log_message(f"[project_apg]: v1_normalized norm: {v1_normalized.norm().item():.4f}", "DEBUG", step, sigma)
        script_instance.log_message(f"[project_apg]: v0_parallel norm: {v0_parallel.norm().item():.4f}, v0_orthogonal norm: {v0_orthogonal.norm().item():.4f}", "DEBUG", step, sigma)

    return v0_parallel, v0_orthogonal

def log_tensor_info_via_instance(script_instance, name, tensor, step, sigma, force_log_early_step=True):
    if not script_instance:
        return
    should_log_this_detail = script_instance.force_all_steps_debug_log_if_global_debug_on or \
                             (script_instance._enable_debug_logging_ui and step < 2 and force_log_early_step)
    
    if not should_log_this_detail and not (script_instance._enable_debug_logging_ui and script_instance.force_all_steps_debug_log):
         if not (script_instance._enable_debug_logging_ui and step < 2 and force_log_early_step and script_instance.should_log_debug()):
            if not script_instance.should_log_debug(False):
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
        script_instance.log_message(f"{log_msg_prefix}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, norm={norm_val}, has_nan={has_nan_val}", "DEBUG", step, sigma)
    elif isinstance(tensor, (int, float, bool)):
        script_instance.log_message(f"{log_msg_prefix}: value={tensor}", "DEBUG", step, sigma)
    else:
        script_instance.log_message(f"{log_msg_prefix}: type={type(tensor)}", "DEBUG", step, sigma)

class APGForge(scripts.Script):
    _instance = None # シングルトンインスタンス管理
    
    group = gr.Group(visible=False) # ダミーのGradioグループ要素。UIには表示されないが、内部参照用

    def __init__(self):
        super().__init__()
        if APGForge._instance is None:
            APGForge._instance = self

        self.apg_enabled = False
        self.apg_eta = 0.0
        self.apg_norm_threshold = 5.0
        self.apg_momentum_beta = 0.0
        self._enable_debug_logging_ui = False
        self.force_all_steps_debug_log = False
        self.running_avg = 0
        self.prev_sigma = None
        self._script_name = "APGForge"
        self.force_all_steps_debug_log_if_global_debug_on = False
        self.user_cfg_scale_for_first_step_override = None

        print(f"[{self._script_name}] Initializing script instance...")
        script_callbacks.on_model_loaded(self.on_model_loaded_instance_method)
        script_callbacks.on_script_unloaded(self.on_script_unloaded_instance_method)
        self.log_message("APGForge Script Initialized.", "INFO")

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
        return "Adaptive Projected Guidance (APG) - Forge"

    def show(self, is_img2img):
        # 修正: scripts.AlwaysHidden の代わりに整数値 1 を返す
        return 1 # 統合スクリプトでUIを制御するためHiddenにする

    def ui(self, is_img2img):
        return [] # 統合スクリプトでUIを制御するため空にする

    def elem_id_prefix(self, is_img2img):
        return f"apg_forge_script_{'img2img' if is_img2img else 'txt2img'}"

    def on_model_loaded_instance_method(self, model):
        self._reset_apg_state()
        model_name = "Unknown Model"
        if hasattr(model, 'sd_model_checkpoint') and model.sd_model_checkpoint:
            model_name = os.path.basename(model.sd_model_checkpoint)
        elif hasattr(shared, 'sd_model') and hasattr(shared.sd_model, 'sd_model_checkpoint') and shared.sd_model.sd_model_checkpoint:
             model_name = os.path.basename(shared.sd_model.sd_model_checkpoint)
        self.log_message(f"APG state reset due to model load/change: {model_name}", "INFO")

    def on_script_unloaded_instance_method(self):
        self.log_message("Script unloaded. Removing APGForge._instance.", "INFO")
        if APGForge._instance == self:
            APGForge._instance = None

    def _reset_apg_state(self):
        self.running_avg = 0
        self.prev_sigma = None
        if self._enable_debug_logging_ui:
            self.log_message(f"APG internal state reset: running_avg=0, prev_sigma=None", "DEBUG")

    def process(self, p,
                enable_debug_logging_ui,
                force_all_steps_debug_log_ui,
                apg_enabled,
                apg_eta,
                apg_norm_threshold,
                apg_momentum_beta):
        
        self._enable_debug_logging_ui = enable_debug_logging_ui
        self.force_all_steps_debug_log = force_all_steps_debug_log_ui
        self.force_all_steps_debug_log_if_global_debug_on = self._enable_debug_logging_ui and self.force_all_steps_debug_log
        
        self.log_message(f"APG process() called. Debug Logging UI: {self._enable_debug_logging_ui}, Force All Steps Log Active: {self.force_all_steps_debug_log_if_global_debug_on}", "INFO")

        self.apg_enabled = apg_enabled
        self.apg_eta = apg_eta
        self.apg_norm_threshold = apg_norm_threshold
        self.apg_momentum_beta = apg_momentum_beta
        if self.should_log_debug(True):
            self.log_message(f"APG params captured: enabled={self.apg_enabled}, eta={self.apg_eta:.2f}, norm_thresh={self.apg_norm_threshold:.2f}, beta={self.apg_momentum_beta:.2f}", "DEBUG")

        # infotextは統合スクリプトで管理するため、ここでは設定しない
        # p.extra_generation_params["APG Debug Logging"] = self._enable_debug_logging_ui
        # p.extra_generation_params["APG Force All Steps Log"] = self.force_all_steps_debug_log
        # p.extra_generation_params["APG Enabled"] = self.apg_enabled
        # ...

    def process_before_every_sampling(self, p, *args, **kwargs):
        # 修正: このメソッドは統合スクリプトによって制御されるため、
        # モデルパッチングのロジックを完全に削除する
        if self.should_log_debug():
            self.log_message("APG's process_before_every_sampling called, but it should be handled by integration.", "DEBUG")
        
        # ただし、user_cfg_scale_for_first_step_overrideのセットアップはここに残す
        current_sampling_step = getattr(shared.state, 'sampling_step', 0)
        is_first_step_of_a_pass = (current_sampling_step == 0)

        if is_first_step_of_a_pass:
            self._reset_apg_state()
            if hasattr(p, 'cfg_scale'):
                self.user_cfg_scale_for_first_step_override = p.cfg_scale
                if self.should_log_debug(True):
                    self.log_message(f"Captured user CFG scale for pass: {self.user_cfg_scale_for_first_step_override}", "DEBUG")
            else:
                self.user_cfg_scale_for_first_step_override = None
                if self.should_log_debug(True):
                    self.log_message(f"p.cfg_scale not found. Cannot determine user CFG scale for override.", "WARNING")
        
        # モデルパッチングのロジックは統合スクリプトに移管されたため、ここから削除
        # unet = p.sd_model.forge_objects.unet.clone() # このcloneも不要になる
        # unet.set_model_sampler_cfg_function(self.apg_cfg_function) # 統合スクリプトが呼ぶ
        # p.sd_model.forge_objects.unet = unet


    @torch.inference_mode() # <-- ここにデコレータを追加 (元からあるが念のため)
    def apg_cfg_function(self, args):
        # print("--- APG_CFG_FUNCTION ENTERED ---") # 無条件プリントはログ機構に任せる


        step = getattr(shared.state, 'sampling_step', -1)
        sigma_tensor_arg = args.get("sigma", torch.tensor([-1.0]))
        if not isinstance(sigma_tensor_arg, torch.Tensor):
            sigma_tensor = torch.tensor([sigma_tensor_arg], device=args["input"].device, dtype=args["input"].dtype)
        else:
            sigma_tensor = sigma_tensor_arg.to(args["input"].device, dtype=args["input"].dtype)


        sigma_val = sigma_tensor[0].item() if sigma_tensor.numel() == 1 else -1.0


        should_log_details_for_this_step = self.should_log_debug(step < 2 or self.force_all_steps_debug_log_if_global_debug_on)


        if should_log_details_for_this_step:
            self.log_message(f"apg_cfg_function invoked.", "DEBUG", step, sigma_val)
        
        cond_scale_from_sampler = args["cond_scale"]
        effective_cond_scale = cond_scale_from_sampler


        is_true_first_step_of_pass = (step == 0 and self.prev_sigma is None) # ここは`is_first_step_of_a_pass`に統合するべきかも
        # APG Forgeのコードでは`is_first_step_of_a_pass`と`current_sampling_step == 0`で判断している。
        # _integrated_cfg_function内で_reset_apg_stateは呼ばれないので、prev_sigmaはそのまま残る。
        # そのため、`is_true_first_step_of_pass`の判定は`process_before_every_sampling`で行われ、
        # `user_cfg_scale_for_first_step_override`が設定される。
        # ここでは単にそれを参照するだけで良い。


        if (step == 0 or (self.prev_sigma is None and sigma_val != -1.0)) and \
           cond_scale_from_sampler == 1.0 and \
           self.user_cfg_scale_for_first_step_override is not None and \
           self.user_cfg_scale_for_first_step_override != 1.0:
            
            effective_cond_scale = self.user_cfg_scale_for_first_step_override
            if self.should_log_debug(True):
                self.log_message(f"FIRST STEP CFG OVERRIDE: Sampler CFG was {cond_scale_from_sampler:.2f}, using stored user CFG {effective_cond_scale:.2f}.", "DEBUG", step, sigma_val)
        
        if should_log_details_for_this_step:
            log_tensor_info_via_instance(self, "args['cond_denoised'] (x0_cond)", args["cond_denoised"], step, sigma_val, force_log_early_step=False)
            log_tensor_info_via_instance(self, "args['uncond_denoised'] (x0_uncond)", args["uncond_denoised"], step, sigma_val, force_log_early_step=False)
            log_tensor_info_via_instance(self, "args['input'] (x_t)", args["input"], step, sigma_val, force_log_early_step=False)
            self.log_message(f"  Cond Scale (w_sampler): {cond_scale_from_sampler:.2f}, Cond Scale (w_eff): {effective_cond_scale:.2f}, Eta (η): {self.apg_eta:.2f}, Beta (β): {self.apg_momentum_beta:.2f}, NormTh (r): {self.apg_norm_threshold:.2f}", "DEBUG", step, sigma_val)




        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]
        x_input = args["input"]


        # Momentum reset logic based on sigma progression or true first step
        # ここでの_reset_apg_state()はprocess_before_every_samplingで呼ばれるため、ここでは不要だが、
        # self.prev_sigmaやself.running_avgの管理はapg_cfg_function自身で行う。
        # APG Forgeのコードでは`_reset_apg_state`が`is_first_step_of_a_pass`で呼ばれるので、
        # `running_avg`と`prev_sigma`は正しくリセットされるはず。
        if self.prev_sigma is not None and sigma_val != -1.0 and self.prev_sigma != -1.0 and sigma_val > self.prev_sigma :
            self.running_avg = 0 
            if should_log_details_for_this_step:
                 self.log_message(f"Momentum reset: sigma increased from {self.prev_sigma:.4f} to {sigma_val:.4f}.", "DEBUG", step, sigma_val)
        
        # is_true_first_step_of_passの判定は`process_before_every_sampling`に任せるため、
        # ここで再度`running_avg = 0`としない。
        # ただし、`process_before_every_sampling`が呼ばれるタイミングはサンプラーのパスの最初なので、
        # `apg_cfg_function`が呼ばれる前に`_reset_apg_state`は確実に実行される。
        # そのため、`is_true_first_step_of_pass`のチェックは不要。


        if sigma_val != -1.0 : self.prev_sigma = sigma_val


        guidance = cond_denoised - uncond_denoised
        if should_log_details_for_this_step:
            log_tensor_info_via_instance(self, "Initial Guidance (ΔD_t)", guidance, step, sigma_val, force_log_early_step=False)


        if self.apg_momentum_beta != 0.0:
            if not torch.is_tensor(self.running_avg) or \
               (isinstance(self.running_avg, torch.Tensor) and self.running_avg.numel() == 1 and self.running_avg.item() == 0) or \
               (isinstance(self.running_avg, (int, float)) and self.running_avg == 0):
                self.running_avg = torch.zeros_like(guidance)
                if should_log_details_for_this_step:
                    self.log_message(f"Momentum: running_avg initialized to zeros_like guidance.", "DEBUG", step, sigma_val)
            
            if self.running_avg.device != guidance.device: self.running_avg = self.running_avg.to(guidance.device)
            if self.running_avg.dtype != guidance.dtype: self.running_avg = self.running_avg.to(guidance.dtype)
            
            self.running_avg = guidance + self.apg_momentum_beta * self.running_avg
            guidance = self.running_avg.clone()
            if should_log_details_for_this_step:
                log_tensor_info_via_instance(self, "Guidance after Momentum (ΔD_t')", guidance, step, sigma_val, force_log_early_step=False)
        elif torch.is_tensor(self.running_avg) and self.running_avg.abs().sum() != 0:
            self.running_avg = 0
            if should_log_details_for_this_step:
                self.log_message("Momentum is 0, non-zero tensor running_avg reset to scalar 0.", "DEBUG", step, sigma_val)


        if self.apg_norm_threshold > 0.0:
            dims_to_operate_norm = [d for d in range(1, guidance.ndim)]
            if not dims_to_operate_norm: dims_to_operate_norm = [0]
            
            guidance_norm = guidance.norm(p=2, dim=dims_to_operate_norm, keepdim=True)
            if should_log_details_for_this_step:
                log_value = guidance_norm.mean().item() if guidance_norm.numel() > 0 else 0.0
                log_tensor_info_via_instance(self, "  Guidance Norm before Clip (mean)", torch.tensor(log_value), step, sigma_val, force_log_early_step=False)


            if (guidance_norm > 1e-8).any():
                scale_factor = torch.minimum(
                    torch.ones_like(guidance_norm, device=guidance.device),
                    self.apg_norm_threshold / (guidance_norm + 1e-8)
                )
                guidance = guidance * scale_factor
                if should_log_details_for_this_step:
                    log_tensor_info_via_instance(self, "Guidance after Norm Clipping (ΔD_t'')", guidance, step, sigma_val, force_log_early_step=False)
                    if isinstance(scale_factor, torch.Tensor) and scale_factor.numel() > 0:
                         log_value_sf = scale_factor.mean().item()
                         log_tensor_info_via_instance(self, "  Norm Clip Scale Factor (mean)", torch.tensor(log_value_sf), step, sigma_val, force_log_early_step=False)
            elif should_log_details_for_this_step:
                self.log_message("  Guidance norm is near zero for all items, skipping norm clipping.", "DEBUG", step, sigma_val)


        guidance_parallel, guidance_orthogonal = project_apg(guidance, cond_denoised, script_instance=self, step=step, sigma=sigma_val)
        if should_log_details_for_this_step:
            log_tensor_info_via_instance(self, "Guidance Parallel (ΔD_t'')_||", guidance_parallel, step, sigma_val, force_log_early_step=False)
            log_tensor_info_via_instance(self, "Guidance Orthogonal (ΔD_t'')_⊥", guidance_orthogonal, step, sigma_val, force_log_early_step=False)


        modified_apg_guidance = guidance_orthogonal + self.apg_eta * guidance_parallel
        if should_log_details_for_this_step:
            log_tensor_info_via_instance(self, "Modified APG Guidance (ΔD_t^{APG})", modified_apg_guidance, step, sigma_val, force_log_early_step=False)


        final_denoised_prediction_apg = uncond_denoised + modified_apg_guidance * effective_cond_scale
        if should_log_details_for_this_step:
            log_tensor_info_via_instance(self, "Final Denoised Prediction with APG (x0_pred_apg)", final_denoised_prediction_apg, step, sigma_val, force_log_early_step=False)


        noise_prediction = x_input - final_denoised_prediction_apg
        if should_log_details_for_this_step:
            log_tensor_info_via_instance(self, "Final Noise Prediction (x_t - x0_pred_apg)", noise_prediction, step, sigma_val, force_log_early_step=False)


        return noise_prediction

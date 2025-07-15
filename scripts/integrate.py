import gradio as gr
import torch
import torch.nn.functional as F

from modules import scripts, shared, script_callbacks
from modules.infotext_utils import PasteField
from modules.shared import opts 

# 既存の拡張機能をインポート
# my_integrated_utils/modules ディレクトリに配置されていることを前提とします
try:
    from my_integrated_utils.modules.APG import APGForge, project_apg, log_tensor_info_via_instance
    from my_integrated_utils.modules.forge_tcfg import TCFGForge
    from my_integrated_utils.modules.mahiro import ScriptMahiro
except ImportError as e:
    print(f"Error importing integrated modules: {e}")
    print("Please ensure APG.py, forge_tcfg.py, and mahiro.py are in 'my_integrated_utils/modules/'")
    APGForge = None
    TCFGForge = None
    ScriptMahiro = None


class IntegratedGuidanceScript(scripts.Script):
    section = "guidance_integrations"
    create_group = True # グループ化してUIを整理
    sorting_priority = 0 # 最も上部に表示されるように設定

    _apg_instance = None
    _tcfg_instance = None
    _mahiro_instance = None

    def __init__(self):
        super().__init__()
        print("IntegratedGuidanceScript: __init__ started.")
        # __init__ メソッド内で各サブスクリプトのシングルトンインスタンスを初期化
        if APGForge and not IntegratedGuidanceScript._apg_instance:
            IntegratedGuidanceScript._apg_instance = APGForge()
            print("IntegratedGuidanceScript: APGForge instance created.")
        if TCFGForge and not IntegratedGuidanceScript._tcfg_instance:
            IntegratedGuidanceScript._tcfg_instance = TCFGForge()
            print("IntegratedGuidanceScript: TCFGForge instance created.")
        if ScriptMahiro and not IntegratedGuidanceScript._mahiro_instance:
            IntegratedGuidanceScript._mahiro_instance = ScriptMahiro()
            print("IntegratedGuidanceScript: ScriptMahiro instance created.")
        print("IntegratedGuidanceScript: __init__ finished.")


    def title(self):
        return "Integrated Guidance (APG + TCFG + MaHiRo)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        elem_id_prefix = f"integrated_guidance_{'img2img' if is_img2img else 'txt2img'}"

        with gr.Accordion(self.title(), open=False, elem_id=f"{elem_id_prefix}_accordion"):
            gr.Markdown("### Core Integration Settings")
            apg_enabled = gr.Checkbox(
                value=False,
                label="Enable Adaptive Projected Guidance (APG) and Integrations",
                elem_id=f"{elem_id_prefix}_apg_enable",
                scale=1
            )
            gr.Markdown(
                "APGを有効にすると、TCFGとMaHiRoも自動的に有効になり、UIに表示されます。"
                "APGを無効にすると、TCFGとMaHiRoも自動的に無効になり、UIから非表示になります。"
            )

            gr.Markdown("---")
            gr.Markdown("### APG Specific Settings (Appears if APG is enabled)")
            apg_eta = gr.Slider(
                label="APG Eta (Parallel Component Scale)",
                minimum=0.0, maximum=1.0, step=0.01, value=0.3,
                elem_id=f"{elem_id_prefix}_apg_eta", visible=False
            )
            apg_norm_threshold = gr.Slider(
                label="APG Norm Threshold (Rescaling Radius r, 0 = disable rescaling)",
                minimum=0.0, maximum=50.0, step=0.1, value=10.0,
                elem_id=f"{elem_id_prefix}_apg_norm_threshold", visible=False
            )
            apg_momentum_beta = gr.Slider(
                label="APG Beta (Momentum Strength, negative for reverse)",
                minimum=-1.0, maximum=1.0, step=0.01, value=-0.15,
                elem_id=f"{elem_id_prefix}_apg_momentum_beta", visible=False
            )
            apg_debug_logging = gr.Checkbox(
                label="APG Debug Logging (Console)",
                value=False,
                elem_id=f"{elem_id_prefix}_apg_debug_logging", visible=False
            )
            apg_force_all_steps_debug_log = gr.Checkbox(
                label="APG Force All Steps Debug Log (Very Verbose)",
                value=False,
                elem_id=f"{elem_id_prefix}_apg_force_all_steps_debug_log", visible=False
            )

            gr.Markdown("---")
            gr.Markdown("### TCFG Specific Settings (Appears if APG is enabled)")
            tcfg_debug_logging = gr.Checkbox(
                label="TCFG Debug Logging (Console)",
                value=False,
                elem_id=f"{elem_id_prefix}_tcfg_debug_logging", visible=False
            )
            tcfg_force_all_steps_debug_log = gr.Checkbox(
                label="TCFG Force All Steps Debug Log (Very Verbose)",
                value=False,
                elem_id=f"{elem_id_prefix}_tcfg_force_all_steps_debug_log", visible=False
            )

            gr.Markdown("---")
            gr.Markdown("### MaHiRo Specific Settings (Appears if APG is enabled)")
            mahiro_debug_logging = gr.Checkbox(
                label="MaHiRo Debug Logging (Console)",
                value=False,
                elem_id=f"{elem_id_prefix}_mahiro_debug_logging", visible=False
            )

        apg_enabled.change(
            fn=lambda x: [
                gr.update(visible=x), gr.update(visible=x), gr.update(visible=x),
                gr.update(visible=x), gr.update(visible=x),
                gr.update(visible=x), gr.update(visible=x),
                gr.update(visible=x)
            ],
            inputs=[apg_enabled],
            outputs=[
                apg_eta, apg_norm_threshold, apg_momentum_beta,
                apg_debug_logging, apg_force_all_steps_debug_log,
                tcfg_debug_logging, tcfg_force_all_steps_debug_log,
                mahiro_debug_logging
            ]
        )

        self.infotext_fields = [
            (apg_enabled, "Integrated Guidance Enabled"),
            (apg_eta, "APG Eta"),
            (apg_norm_threshold, "APG Norm Threshold"),
            (apg_momentum_beta, "APG Beta"),
            (apg_debug_logging, "APG Debug Logging"),
            (apg_force_all_steps_debug_log, "APG Force All Steps Log"),
            (tcfg_debug_logging, "TCFG Debug Logging"),
            (tcfg_force_all_steps_debug_log, "TCFG Force All Steps Log"),
            (mahiro_debug_logging, "MaHiRo Debug Logging"),
        ]
        return [
            apg_enabled,
            apg_eta, apg_norm_threshold, apg_momentum_beta,
            apg_debug_logging, apg_force_all_steps_debug_log,
            tcfg_debug_logging, tcfg_force_all_steps_debug_log,
            mahiro_debug_logging
        ]

    def process(self, p,
                apg_enabled,
                apg_eta, apg_norm_threshold, apg_momentum_beta,
                apg_debug_logging, apg_force_all_steps_debug_log,
                tcfg_debug_logging, tcfg_force_all_steps_debug_log,
                mahiro_debug_logging):
        
        if self._apg_instance:
            self._apg_instance.process(
                p, apg_debug_logging, apg_force_all_steps_debug_log, apg_enabled,
                apg_eta, apg_norm_threshold, apg_momentum_beta
            )
            if apg_enabled:
                if self._tcfg_instance:
                    self._tcfg_instance.process(p, tcfg_debug_logging, tcfg_force_all_steps_debug_log, True)
                if self._mahiro_instance:
                    self._mahiro_instance.process(p, True, mahiro_debug_logging)
            else:
                if self._tcfg_instance:
                    self._tcfg_instance.process(p, False, False, False)
                if self._mahiro_instance:
                    self._mahiro_instance.process(p, False, False)

        p.extra_generation_params["Integrated Guidance Enabled"] = apg_enabled
        if apg_enabled:
            p.extra_generation_params["APG Eta"] = f"{apg_eta:.2f}"
            p.extra_generation_params["APG Norm Threshold"] = f"{apg_norm_threshold:.2f}"
            p.extra_generation_params["APG Beta"] = f"{apg_momentum_beta:.2f}"
            p.extra_generation_params["TCFG Enabled (Auto)"] = True
            p.extra_generation_params["MaHiRo Enabled (Auto)"] = True
        
    def process_before_every_sampling(self, p, *args, **kwargs):
        # ーーー 修正点: 状態リセットロジックを追加 ーーー
        current_sampling_step = getattr(shared.state, 'sampling_step', 0)
        if current_sampling_step == 0:
            if self._apg_instance:
                self._apg_instance._reset_apg_state()
                if hasattr(p, 'cfg_scale'):
                    self._apg_instance.user_cfg_scale_for_first_step_override = p.cfg_scale
            if self._tcfg_instance:
                self._tcfg_instance._reset_tcfg_state()
                if hasattr(p, 'cfg_scale'):
                    self._tcfg_instance.user_cfg_scale_for_first_step_override = p.cfg_scale
        # ーーー 修正ここまで ーーー

        if not self._apg_instance:
            print("IntegratedGuidanceScript: APG instance not initialized, skipping hooks.")
            return

        apg_is_currently_enabled = self._apg_instance.apg_enabled

        model_patcher = None
        if hasattr(shared, 'sd_model') and hasattr(shared.sd_model, 'forge_objects') and hasattr(shared.sd_model.forge_objects, 'unet'):
            model_patcher_candidate = shared.sd_model.forge_objects.unet
            if hasattr(model_patcher_candidate, 'set_model_sampler_cfg_function'):
                model_patcher = model_patcher_candidate
            elif hasattr(model_patcher_candidate, 'model') and hasattr(model_patcher_candidate.model, 'set_model_sampler_cfg_function'):
                model_patcher = model_patcher_candidate.model
        
        if not model_patcher:
            print("IntegratedGuidanceScript: Could not find ModelPatcher, CFG functions will not be hooked.")
            return

        self._apg_instance.log_message(f"process_before_every_sampling called. Integrated guidance enabled: {apg_is_currently_enabled}", "INFO")

        if apg_is_currently_enabled:
            model_patcher.set_model_sampler_cfg_function(self._integrated_cfg_function)
            self._apg_instance.log_message("Integrated guidance CFG function hooked.", "INFO")
        else:
            model_patcher.set_model_sampler_cfg_function(self._default_cfg_function)
            self._apg_instance.log_message("Integrated guidance disabled. Reverted to default CFG function.", "INFO")
    
    @torch.inference_mode()
    def _default_cfg_function(self, args: dict):
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]
        cond_scale = args["cond_scale"]
        x_input = args["input"]
        denoised = uncond_denoised + (cond_denoised - uncond_denoised) * cond_scale
        noise_pred = x_input - denoised
        return noise_pred
    
    # ーーー 修正点: ガイダンス計算のロジックを全面的に修正 ーーー
    @torch.inference_mode()
    def _integrated_cfg_function(self, args: dict):
        """
        TCFG -> APG -> MaHiRo の順で適用される統合されたCFG関数
        """
        step = getattr(shared.state, 'sampling_step', -1)
        sigma_tensor_arg = args.get("sigma", torch.tensor([-1.0]))
        if not isinstance(sigma_tensor_arg, torch.Tensor):
            sigma_tensor = torch.tensor([sigma_tensor_arg], device=args["input"].device, dtype=args["input"].dtype)
        else:
            sigma_tensor = sigma_tensor_arg.to(args["input"].device, dtype=args["input"].dtype)
        sigma_val = sigma_tensor[0].item() if sigma_tensor.numel() == 1 else -1.0

        apg_script_instance = self._apg_instance
        tcfg_script_instance = self._tcfg_instance
        mahiro_script_instance = self._mahiro_instance

        # 初期テンソルを取得
        current_x_input = args["input"]
        cond_denoised_initial = args["cond_denoised"]
        uncond_denoised_initial = args["uncond_denoised"]
        initial_cond_scale = args["cond_scale"]

        should_log_details = apg_script_instance.should_log_debug(step < 2 or apg_script_instance.force_all_steps_debug_log_if_global_debug_on)
        if should_log_details:
            apg_script_instance.log_message(f"--- INTEGRATED_CFG_FUNCTION ENTERED (Step: {step}, Sigma: {sigma_val:.4f}) ---", "DEBUG", step, sigma_val)
            log_tensor_info_via_instance(apg_script_instance, "Initial cond_denoised", cond_denoised_initial, step, sigma_val)
            log_tensor_info_via_instance(apg_script_instance, "Initial uncond_denoised", uncond_denoised_initial, step, sigma_val)

        # ----------------------------------------------------
        # 1. TCFGの適用 (uncond_denoisedの修正)
        uncond_denoised_after_tcfg = uncond_denoised_initial
        if tcfg_script_instance and tcfg_script_instance.tcfg_enabled:
            if should_log_details:
                apg_script_instance.log_message(f"--- TCFG Pre-computation for APG ---", "DEBUG", step, sigma_val)

            noise_pred_cond = current_x_input - cond_denoised_initial
            noise_pred_uncond = current_x_input - uncond_denoised_initial

            uncond_td_noise_pred = tcfg_script_instance.score_tangential_damping(noise_pred_cond, noise_pred_uncond, step, sigma_val)
            uncond_denoised_after_tcfg = current_x_input - uncond_td_noise_pred # TCFG適用後のuncond

            if should_log_details:
                log_tensor_info_via_instance(apg_script_instance, "Uncond Denoised after TCFG mod", uncond_denoised_after_tcfg, step, sigma_val)

        # ----------------------------------------------------
        # 2. APGの適用
        # APGが無効な場合は、TCFG適用後の結果（または素のCFG）でノイズ予測を計算する
        denoised_after_guidance = uncond_denoised_after_tcfg + initial_cond_scale * (cond_denoised_initial - uncond_denoised_after_tcfg)

        if apg_script_instance and apg_script_instance.apg_enabled:
            if should_log_details:
                apg_script_instance.log_message(f"--- APG Processing ---", "DEBUG", step, sigma_val)

            apg_args = {
                "input": current_x_input,
                "cond_denoised": cond_denoised_initial,
                "uncond_denoised": uncond_denoised_after_tcfg,
                "cond_scale": initial_cond_scale,
                "sigma": sigma_tensor,
            }
            
            # APG関数は内部でガイダンス計算から最終的なノイズ予測までを行う
            noise_pred_after_apg = apg_script_instance.apg_cfg_function(apg_args)
            denoised_after_guidance = current_x_input - noise_pred_after_apg
            
            if should_log_details:
                log_tensor_info_via_instance(apg_script_instance, "Denoised After APG (incl. TCFG)", denoised_after_guidance, step, sigma_val)
        
        # ----------------------------------------------------
        # 3. MaHiRoの適用
        final_denoised_prediction = denoised_after_guidance
        if mahiro_script_instance and mahiro_script_instance.mahiro_enabled:
            if should_log_details:
                apg_script_instance.log_message(f"--- MaHiRo Processing ---", "DEBUG", step, sigma_val)

            mahiro_args = {
                "cond_scale": initial_cond_scale,
                "cond_denoised": cond_denoised_initial,
                "uncond_denoised": uncond_denoised_initial,
                "denoised": denoised_after_guidance,
                "sigma": sigma_tensor
            }
            final_denoised_prediction = mahiro_script_instance.mahiro_normd(mahiro_args)

            if should_log_details:
                log_tensor_info_via_instance(apg_script_instance, "Final Denoised After MaHiRo", final_denoised_prediction, step, sigma_val)

        # ----------------------------------------------------
        # 最終的なノイズ予測をサンプラーに返す
        final_noise_prediction_for_sampler = current_x_input - final_denoised_prediction

        if should_log_details:
            log_tensor_info_via_instance(apg_script_instance, "--- FINAL Noise Prediction for Sampler ---", final_noise_prediction_for_sampler, step, sigma_val)
            apg_script_instance.log_message(f"--- INTEGRATED_CFG_FUNCTION EXITED (Step: {step}, Sigma: {sigma_val:.4f}) ---", "DEBUG", step, sigma_val)

        return final_noise_prediction_for_sampler

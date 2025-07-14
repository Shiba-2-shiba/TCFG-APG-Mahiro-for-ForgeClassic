import gradio as gr
import torch
import torch.nn.functional as F

from modules import scripts, shared, script_callbacks
from modules.infotext_utils import PasteField
from modules.shared import opts # optsはMaHiRoの古い参照を削除したため、基本的には不要だが、念のため残す

# 既存の拡張機能をインポート
# my_integrated_utils.modules ディレクトリに配置されていることを前提とします
try:
    from my_integrated_utils.modules.APG import APGForge, project_apg, log_tensor_info_via_instance
    from my_integrated_utils.modules.forge_tcfg import TCFGForge
    from my_integrated_utils.modules.mahiro import ScriptMahiro
except ImportError as e:
    print(f"Error importing integrated modules: {e}")
    print("Please ensure APG.py, forge_tcfg.py, and mahiro.py are in 'my_integrated_utils/modules/'")
    # インポートエラーが発生した場合、スクリプトの動作を停止させるか、フォールバックを提供することを検討
    # ここでは単純にエラーメッセージを出力し、後続の処理でNoneチェックを行う
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
        print("IntegratedGuidanceScript: __init__ started.") # デバッグログ
        # 修正: __init__ メソッド内で各サブスクリプトのシングルトンインスタンスを初期化
        # これにより、process_before_every_sampling が呼び出される前にインスタンスが確実に存在する
        if APGForge and not IntegratedGuidanceScript._apg_instance:
            IntegratedGuidanceScript._apg_instance = APGForge()
            print("IntegratedGuidanceScript: APGForge instance created.") # デバッグログ
        if TCFGForge and not IntegratedGuidanceScript._tcfg_instance:
            IntegratedGuidanceScript._tcfg_instance = TCFGForge()
            print("IntegratedGuidanceScript: TCFGForge instance created.") # デバッグログ
        if ScriptMahiro and not IntegratedGuidanceScript._mahiro_instance:
            IntegratedGuidanceScript._mahiro_instance = ScriptMahiro()
            print("IntegratedGuidanceScript: ScriptMahiro instance created.") # デバッグログ
        print("IntegratedGuidanceScript: __init__ finished.") # デバッグログ


    def title(self):
        return "Integrated Guidance (APG + TCFG + MaHiRo)"

    def show(self, is_img2img):
        # 常に表示し、UIのチェックボックスで有効/無効を制御
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # UI要素のIDにユニークなプレフィックスを付与
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
            # APGのUI要素。APG有効時に表示を制御するため、初期値はFalse/0
            apg_eta = gr.Slider(
                label="APG Eta (Parallel Component Scale)",
                minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                elem_id=f"{elem_id_prefix}_apg_eta", visible=False # 初期は非表示
            )
            apg_norm_threshold = gr.Slider(
                label="APG Norm Threshold (Rescaling Radius r, 0 = disable rescaling)",
                minimum=0.0, maximum=50.0, step=0.1, value=5.0, # 論文推奨値に近い初期値
                elem_id=f"{elem_id_prefix}_apg_norm_threshold", visible=False # 初期は非表示
            )
            apg_momentum_beta = gr.Slider(
                label="APG Beta (Momentum Strength, negative for reverse)",
                minimum=-1.0, maximum=1.0, step=0.01, value=-0.5, # 論文推奨値に近い初期値
                elem_id=f"{elem_id_prefix}_apg_momentum_beta", visible=False # 初期は非表示
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
            # TCFGのUI要素
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
            # MaHiRoのUI要素
            mahiro_debug_logging = gr.Checkbox(
                label="MaHiRo Debug Logging (Console)",
                value=False,
                elem_id=f"{elem_id_prefix}_mahiro_debug_logging", visible=False
            )

        # APG有効/無効に応じてTCFGとMaHiRoのUI表示を切り替えるロジック
        apg_enabled.change(
            fn=lambda x: [
                gr.update(visible=x), gr.update(visible=x), gr.update(visible=x),
                gr.update(visible=x), gr.update(visible=x), # APG debug
                gr.update(visible=x), gr.update(visible=x), # TCFG debug
                gr.update(visible=x) # MaHiRo debug
            ],
            inputs=[apg_enabled],
            outputs=[
                apg_eta, apg_norm_threshold, apg_momentum_beta,
                apg_debug_logging, apg_force_all_steps_debug_log,
                tcfg_debug_logging, tcfg_force_all_steps_debug_log,
                mahiro_debug_logging
            ]
        )

        # Infotext fields (保存・読み込み用)
        self.infotext_fields = [
            (apg_enabled, "Integrated Guidance Enabled"), # 統合のメインスイッチ
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
        
        # process()が呼ばれるたびに、最新のUIパラメータをサブスクリプトインスタンスに渡す
        if IntegratedGuidanceScript._apg_instance:
            IntegratedGuidanceScript._apg_instance.process(
                p, apg_debug_logging, apg_force_all_steps_debug_log, apg_enabled,
                apg_eta, apg_norm_threshold, apg_momentum_beta
            )
            # APGが有効な場合、TCFGとMaHiRoも自動的に有効にする
            if apg_enabled:
                if IntegratedGuidanceScript._tcfg_instance:
                    IntegratedGuidanceScript._tcfg_instance.process(
                        p, tcfg_debug_logging, tcfg_force_all_steps_debug_log, True # TCFGを強制的に有効化
                    )
                if IntegratedGuidanceScript._mahiro_instance:
                    # MaHiRoのprocessはenableチェックボックスとdebug loggingを受け取る想定
                    IntegratedGuidanceScript._mahiro_instance.process(
                        p, True, mahiro_debug_logging # MaHiRoを強制的に有効化、デバッグログも渡す
                    )
            else:
                # APGが無効の場合、TCFGとMaHiRoも強制的に無効にする
                if IntegratedGuidanceScript._tcfg_instance:
                    IntegratedGuidanceScript._tcfg_instance.process(
                        p, tcfg_debug_logging, tcfg_force_all_steps_debug_log, False # TCFGを強制的に無効化
                    )
                if IntegratedGuidanceScript._mahiro_instance:
                    IntegratedGuidanceScript._mahiro_instance.process(
                        p, False, mahiro_debug_logging # MaHiRoを強制的に無効化
                    )

        # Infotextの更新
        p.extra_generation_params["Integrated Guidance Enabled"] = apg_enabled
        if apg_enabled:
            p.extra_generation_params["APG Eta"] = f"{apg_eta:.2f}"
            p.extra_generation_params["APG Norm Threshold"] = f"{apg_norm_threshold:.2f}"
            p.extra_generation_params["APG Beta"] = f"{apg_momentum_beta:.2f}"
            p.extra_generation_params["APG Debug Logging"] = apg_debug_logging
            p.extra_generation_params["APG Force All Steps Log"] = apg_force_all_steps_debug_log
            p.extra_generation_params["TCFG Enabled (Auto)"] = True
            p.extra_generation_params["TCFG Debug Logging"] = tcfg_debug_logging
            p.extra_generation_params["TCFG Force All Steps Log"] = tcfg_force_all_steps_debug_log
            p.extra_generation_params["MaHiRo Enabled (Auto)"] = True
            p.extra_generation_params["MaHiRo Debug Logging"] = mahiro_debug_logging
        else:
            p.extra_generation_params["APG Eta"] = "N/A (Disabled)"
            p.extra_generation_params["APG Norm Threshold"] = "N/A (Disabled)"
            p.extra_generation_params["APG Beta"] = "N/A (Disabled)"
            p.extra_generation_params["APG Debug Logging"] = False
            p.extra_generation_params["APG Force All Steps Log"] = False
            p.extra_generation_params["TCFG Enabled (Auto)"] = False
            p.extra_generation_params["TCFG Debug Logging"] = False
            p.extra_generation_params["TCFG Force All Steps Log"] = False
            p.extra_generation_params["MaHiRo Enabled (Auto)"] = False
            p.extra_generation_params["MaHiRo Debug Logging"] = False


    def process_before_every_sampling(self, p, *args, **kwargs):
        # 修正: __init__ でインスタンスが初期化されるため、ここではNoneチェックは不要だが、
        # 万が一のフォールバックとして残しても害はない
        if not IntegratedGuidanceScript._apg_instance:
            print("IntegratedGuidanceScript: APG instance not initialized in __init__, skipping hooks. This should not happen.")
            return

        # UIでAPGが有効にされているか確認 (process_before_every_samplingにはapg_enabledは直接渡されないため、インスタンス変数から取得)
        # ただし、apg_enabledはprocessで設定されるため、_apg_instance.apg_enabledを参照するのが最も正確
        apg_is_currently_enabled = IntegratedGuidanceScript._apg_instance.apg_enabled

        # ModelPatcherインスタンスの取得ロジックはそのまま
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

        # シングルトンインスタンスの_apg_instanceからlog_messageを呼び出す
        IntegratedGuidanceScript._apg_instance.log_message(f"process_before_every_sampling called. Integrated guidance enabled: {apg_is_currently_enabled}", "INFO")

        if apg_is_currently_enabled:
            # 統合されたカスタムCFG関数をセット
            model_patcher.set_model_sampler_cfg_function(self._integrated_cfg_function)
            IntegratedGuidanceScript._apg_instance.log_message("Integrated guidance CFG function hooked.", "INFO")
        else:
            # APGが無効の場合、標準のCFG関数をセットしてフックを実質的に解除
            model_patcher.set_model_sampler_cfg_function(self._default_cfg_function)
            IntegratedGuidanceScript._apg_instance.log_message("Integrated guidance disabled. Reverted to default CFG function.", "INFO")
    
    @torch.inference_mode()
    def _default_cfg_function(self, args: dict):
        """
        標準のCFG動作を再現し、カスタム関数を安全に解除するための関数。
        最終的なノイズ予測を返す。
        """
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]
        cond_scale = args["cond_scale"]
        x_input = args["input"]

        # 標準的なCFGの計算
        denoised = uncond_denoised + (cond_denoised - uncond_denoised) * cond_scale

        # サンプラーはデノイズされた画像ではなく、ノイズ予測を期待する
        noise_pred = x_input - denoised

        return noise_pred
    
    
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

        apg_script_instance = IntegratedGuidanceScript._apg_instance
        tcfg_script_instance = IntegratedGuidanceScript._tcfg_instance
        mahiro_script_instance = IntegratedGuidanceScript._mahiro_instance

        # デバッグログの条件を決定
        should_log_details = apg_script_instance.should_log_debug(step < 2 or apg_script_instance.force_all_steps_debug_log_if_global_debug_on)
        if should_log_details:
            apg_script_instance.log_message(f"--- INTEGRATED_CFG_FUNCTION ENTERED (Step: {step}, Sigma: {sigma_val:.4f}) ---", "DEBUG", step, sigma_val)
            log_tensor_info_via_instance(apg_script_instance, "Initial cond_denoised", args["cond_denoised"], step, sigma_val)
            log_tensor_info_via_instance(apg_script_instance, "Initial uncond_denoised", args["uncond_denoised"], step, sigma_val)
            log_tensor_info_via_instance(apg_script_instance, "Initial input (x_t)", args["input"], step, sigma_val)

        # 最初のノイズ予測を生成 (CFGスケール1.0でのuncond_denoisedとcond_denoisedから)
        cond_denoised_initial = args["cond_denoised"]
        uncond_denoised_initial = args["uncond_denoised"]
        current_x_input = args["input"] # x_t
        initial_cond_scale = args["cond_scale"] # ユーザーが設定したCFGスケールw

        # ----------------------------------------------------
        # 1. TCFGの適用
        # TCFGはノイズ予測を期待するので、denoised_initialからノイズ予測を逆算
        noise_pred_cond = current_x_input - cond_denoised_initial
        noise_pred_uncond = current_x_input - uncond_denoised_initial

        final_noise_pred_after_tcfg = noise_pred_uncond # 初期値
        
        if tcfg_script_instance and tcfg_script_instance.tcfg_enabled:
            if should_log_details:
                apg_script_instance.log_message(f"--- TCFG Processing ---", "DEBUG", step, sigma_val)
                log_tensor_info_via_instance(apg_script_instance, "Noise Pred Cond (Pre-TCFG)", noise_pred_cond, step, sigma_val)
                log_tensor_info_via_instance(apg_script_instance, "Noise Pred Uncond (Pre-TCFG)", noise_pred_uncond, step, sigma_val)

            # TCFGの`tcfg_cfg_function`を直接呼び出し、必要な引数を渡す
            # `tcfg_cfg_function`は`args` dict全体を期待し、内部でCFGスケールオーバーライドロジックを処理する
            # NOTE: TCFGのtcfg_cfg_functionはargsを直接受け取るため、元のcond_scaleを一時的に変更する必要はありません。
            # TCFG内部でuser_cfg_scale_for_first_step_overrideが適用されます。
            final_noise_pred_after_tcfg = tcfg_script_instance.tcfg_cfg_function(args)
            
            if should_log_details:
                log_tensor_info_via_instance(apg_script_instance, "Final Noise Pred After TCFG", final_noise_pred_after_tcfg, step, sigma_val)
        else:
            # TCFGが無効な場合、通常のCFGのノイズ予測を使用
            final_noise_pred_after_tcfg = noise_pred_uncond + initial_cond_scale * (noise_pred_cond - noise_pred_uncond)
            if should_log_details:
                apg_script_instance.log_message(f"TCFG disabled, using standard CFG noise pred.", "DEBUG", step, sigma_val)

        # TCFGの結果は最終的なノイズ予測なので、これを使ってデノイズ済み画像を逆算
        denoised_after_tcfg = current_x_input - final_noise_pred_after_tcfg
        
        if should_log_details:
            log_tensor_info_via_instance(apg_script_instance, "Denoised After TCFG (x0_tcfg)", denoised_after_tcfg, step, sigma_val)
        

        # ----------------------------------------------------
        # 2. APGの適用
        # APGはcond_denoisedとuncond_denoisedを期待する。
        # TCFG適用後の結果をAPGの条件付き入力として使用し、元のuncond_denoisedを使用
        apg_args = {
            "input": current_x_input,
            "cond_denoised": denoised_after_tcfg, # TCFG適用後の結果をAPGの条件付き入力として使用
            "uncond_denoised": uncond_denoised_initial, # TCFGはuncond_denoisedを修正しないため、元のuncondを使用
            "cond_scale": initial_cond_scale, # APGも内部でCFGスケールを調整するため、元のスケールを渡す
            "sigma": sigma_tensor, # sigmaを渡す
            "denoised": denoised_after_tcfg # APGの`denoised`引数は使われていないようだが、念のため渡す
        }
        
        final_noise_pred_after_apg = final_noise_pred_after_tcfg # 初期値
        
        if apg_script_instance and apg_script_instance.apg_enabled:
            if should_log_details:
                apg_script_instance.log_message(f"--- APG Processing ---", "DEBUG", step, sigma_val)
                log_tensor_info_via_instance(apg_script_instance, "APG Input cond_denoised (from TCFG)", apg_args["cond_denoised"], step, sigma_val)
                log_tensor_info_via_instance(apg_script_instance, "APG Input uncond_denoised (original)", apg_args["uncond_denoised"], step, sigma_val)
                apg_script_instance.log_message(f"APG effective_cond_scale: {apg_args['cond_scale']:.2f}", "DEBUG", step, sigma_val)

            # APGの`apg_cfg_function`を直接呼び出し、必要な引数を渡す
            final_noise_pred_after_apg = apg_script_instance.apg_cfg_function(apg_args)

            if should_log_details:
                log_tensor_info_via_instance(apg_script_instance, "Final Noise Pred After APG", final_noise_pred_after_apg, step, sigma_val)
        else:
            # APGが無効な場合、TCFGの結果をそのまま使用
            final_noise_pred_after_apg = final_noise_pred_after_tcfg
            if should_log_details:
                apg_script_instance.log_message(f"APG disabled, using TCFG result as is.", "DEBUG", step, sigma_val)
            
        # APGの結果は最終的なノイズ予測なので、これを使ってデノイズ済み画像を逆算
        denoised_after_apg = current_x_input - final_noise_pred_after_apg
        
        if should_log_details:
            log_tensor_info_via_instance(apg_script_instance, "Denoised After APG (x0_apg)", denoised_after_apg, step, sigma_val)
        
        # ----------------------------------------------------
        # 3. MaHiRoの適用 (Post-CFG Functionとして機能させる)
        # MaHiRoは`mahiro_normd`関数として実装されており、
        # `cond_denoised`, `uncond_denoised`, `denoised` を引数に取る
        mahiro_args = {
            "cond_scale": initial_cond_scale, # 元のCFGスケール
            "cond_denoised": cond_denoised_initial, # 元の条件付きデノイズ結果
            "uncond_denoised": uncond_denoised_initial, # 元の無条件デノイズ結果
            "denoised": denoised_after_apg, # APGによって修正された最終的なデノイズ済み結果
            "sigma": sigma_tensor # sigmaを渡す
        }

        final_denoised_prediction = denoised_after_apg # 初期値

        # MaHiRoが有効な場合、mahiro_normd関数を呼び出す
        # 修正箇所: opts.show_mahiro の参照を削除し、mahiro_script_instance.mahiro_enabled を使用
        if mahiro_script_instance and mahiro_script_instance.mahiro_enabled: 
            if should_log_details:
                apg_script_instance.log_message(f"--- MaHiRo Processing ---", "DEBUG", step, sigma_val)
                log_tensor_info_via_instance(apg_script_instance, "MaHiRo Input cond_denoised (original)", mahiro_args["cond_denoised"], step, sigma_val)
                log_tensor_info_via_instance(apg_script_instance, "MaHiRo Input uncond_denoised (original)", mahiro_args["uncond_denoised"], step, sigma_val)
                log_tensor_info_via_instance(apg_script_instance, "MaHiRo Input denoised (from APG)", mahiro_args["denoised"], step, sigma_val)
                
            # MaHiRoの関数を直接呼び出す
            final_denoised_prediction = mahiro_script_instance.mahiro_normd(mahiro_args)
            
            if should_log_details:
                log_tensor_info_via_instance(apg_script_instance, "Final Denoised After MaHiRo", final_denoised_prediction, step, sigma_val)
        else:
            final_denoised_prediction = denoised_after_apg
            if should_log_details:
                apg_script_instance.log_message(f"MaHiRo disabled, using APG result as is.", "DEBUG", step, sigma_val)

        # 最終的なデノイズ済み予測からノイズ予測を逆算して返す
        # Forgeの`set_model_sampler_cfg_function`は最終的にノイズ予測を期待する
        final_noise_prediction_for_sampler = current_x_input - final_denoised_prediction

        if should_log_details:
            log_tensor_info_via_instance(apg_script_instance, "--- FINAL Noise Prediction for Sampler ---", final_noise_prediction_for_sampler, step, sigma_val)
            apg_script_instance.log_message(f"--- INTEGRATED_CFG_FUNCTION EXITED (Step: {step}, Sigma: {sigma_val:.4f}) ---", "DEBUG", step, sigma_val)

        return final_noise_prediction_for_sampler
    
    # この部分は、APGが有効な場合にのみ_integrated_cfg_functionがフックされるため、
    # 基本的にデッドコードとなる。
    # return args["input"] - args["denoised"] # デフォルトのCFG結果を返す（APGが無効な場合）

import sys
import os
import argparse
import platform
import warnings
warnings.filterwarnings("ignore")

import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def remove_bg(input_path, output_path, model_name="u2net.onnx", use_gpu=False, edge_shift=0, edge_feather=0):
    try:
        # 1. Hardware Providers
        available_providers = ort.get_available_providers()
        providers = []
        if use_gpu:
            preferred = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CoreMLExecutionProvider', 'ROCMExecutionProvider']
            for p in preferred:
                if p in available_providers:
                    providers.append(p)
        if 'CPUExecutionProvider' in available_providers:
            providers.append('CPUExecutionProvider')
        if not providers:
            providers = ['CPUExecutionProvider']

        # 2. Bulletproof Path Resolution
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(os.path.dirname(sys.executable))
            exe_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            exe_dir = base_dir

        # Check for RMBG-1.4 (SOTA 1024x1024 model)
        rmbg_file = os.path.join(base_dir, "rmbg-1.4.onnx")
        if not os.path.exists(rmbg_file):
            rmbg_file = os.path.join(exe_dir, "rmbg-1.4.onnx")

        # Upgrade default/u2net requests to RMBG-1.4 if available
        if ("u2net" in model_name.lower() or "rmbg" in model_name.lower()) and os.path.exists(rmbg_file):
            model_path = rmbg_file
            is_rmbg = True
            is_isnet = False
        else:
            is_rmbg = False
            if os.path.isabs(model_name) and os.path.exists(model_name):
                model_path = model_name
            elif os.path.exists(os.path.join(os.getcwd(), model_name)):
                model_path = os.path.join(os.getcwd(), model_name)
            elif os.path.exists(os.path.join(base_dir, model_name)):
                model_path = os.path.join(base_dir, model_name)
            elif os.path.exists(os.path.join(exe_dir, model_name)):
                model_path = os.path.join(exe_dir, model_name)
            else:
                model_path = os.path.join(base_dir, model_name)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Cannot find AI model at: {model_path} (checked cwd: {os.getcwd()})")

            is_isnet = "isnet" in os.path.basename(model_path).lower()

        # 3. Speed Optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

        # 4. Input Preprocessing
        img = Image.open(input_path).convert("RGB")
        orig_w, orig_h = img.size
        
        target_size = 1024 if (is_rmbg or is_isnet) else 320
        img_resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 255.0

        if is_rmbg or is_isnet:
            # 1024x1024 SOTA Normalization: (x - 0.5) / 1.0
            img_norm = (img_np - [0.5, 0.5, 0.5]) / [1.0, 1.0, 1.0]
        else:
            # U2Net ImageNet Normalization
            img_norm = (img_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        img_data = np.transpose(img_norm, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)

        # 5. Run Model Inference
        inputs = {session.get_inputs()[0].name: img_data}
        ort_outs = session.run(None, inputs)
        raw_mask = ort_outs[0][0, 0, :, :]

        # Normalize probability map
        m_min = np.min(raw_mask)
        m_max = np.max(raw_mask)
        if m_max > m_min:
            raw_mask = (raw_mask - m_min) / (m_max - m_min)
        raw_mask = np.clip(raw_mask, 0.0, 1.0)

        # 6. Smooth Upscaling to Native Image Resolution
        mask_full = cv2.resize(raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        mask_full = np.clip(mask_full, 0.0, 1.0)

        # 7. Background Screen Analysis (Chroma / Studio Detection)
        rgb = np.array(img).astype(np.float32) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        border_px = max(4, min(orig_h, orig_w) // 50)
        corners = np.concatenate([
            rgb[:border_px, :border_px].reshape(-1, 3),
            rgb[:border_px, -border_px:].reshape(-1, 3),
            rgb[-border_px:, :border_px].reshape(-1, 3),
            rgb[-border_px:, -border_px:].reshape(-1, 3)
        ], axis=0)
        bg_median = np.median(corners, axis=0)

        is_green = bool((bg_median[1] > bg_median[0] + 0.12) and (bg_median[1] > bg_median[2] + 0.12))
        is_blue = bool((bg_median[2] > bg_median[0] + 0.12) and (bg_median[2] > bg_median[1] + 0.12))

        clean_rgb = rgb.copy()

        # 8. Green / Blue Screen Despill & Sub-pixel Chroma Keying
        if is_green:
            max_rb = np.maximum(r, b)
            spill = np.maximum(0.0, g - max_rb)
            # Despill: clamp green bleed to natural adjacent tones
            clean_rgb[:, :, 1] = np.where(spill > 0, np.minimum(g, 0.5 * r + 0.5 * b), g)

            # Keylight analytical chroma matte
            chroma_diff = g - max_rb
            chroma_matte = 1.0 - np.clip((chroma_diff - 0.01) / 0.07, 0.0, 1.0)
            mask_full = np.minimum(mask_full, chroma_matte)

        elif is_blue:
            max_rg = np.maximum(r, g)
            spill = np.maximum(0.0, b - max_rg)
            clean_rgb[:, :, 2] = np.where(spill > 0, np.minimum(b, 0.5 * r + 0.5 * b), b)

            chroma_diff = b - max_rg
            chroma_matte = 1.0 - np.clip((chroma_diff - 0.01) / 0.07, 0.0, 1.0)
            mask_full = np.minimum(mask_full, chroma_matte)

        # 9. Smart Graphic Icon & Enclosed Hole Recovery
        # For vector/graphic icons (like mouse cursors), recover enclosed interior regions
        mask_bin = (mask_full > 0.4).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            h_info = hierarchy[0]
            hole_mask = np.zeros_like(mask_bin)
            has_holes = False
            for i, c in enumerate(contours):
                parent_idx = h_info[i][3]
                if parent_idx != -1:  # Enclosed interior hole
                    area = cv2.contourArea(c)
                    if area > 20: # Non-trivial interior
                        cv2.drawContours(hole_mask, contours, i, 1, thickness=cv2.FILLED)
                        has_holes = True
            if has_holes:
                mask_full = np.maximum(mask_full, hole_mask.astype(np.float32))

        # 10. Clean Faint Outer Haze / Backdrop Noise
        # Gently suppress tiny floating dust in background while protecting the subject
        mask_full = np.where(mask_full < 0.04, 0.0, mask_full)
        mask_full = np.where(mask_full > 0.96, 1.0, mask_full)

        alpha_u8 = (np.clip(mask_full, 0.0, 1.0) * 255).astype(np.uint8)

        # 11. Native Resolution Edge Shift & Feathering
        if edge_shift != 0:
            val = abs(edge_shift)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (val * 2 + 1, val * 2 + 1))
            if edge_shift > 0:
                alpha_u8 = cv2.dilate(alpha_u8, kernel, iterations=1)
            else:
                alpha_u8 = cv2.erode(alpha_u8, kernel, iterations=1)

        if edge_feather > 0:
            k_size = edge_feather * 2 + 1
            alpha_u8 = cv2.GaussianBlur(alpha_u8, (k_size, k_size), 0)

        # 12. Save Production Cutout
        clean_rgb_u8 = (clean_rgb * 255).astype(np.uint8)
        out_rgba = np.dstack([clean_rgb_u8, alpha_u8])
        out_img = Image.fromarray(out_rgba, mode="RGBA")
        out_img.save(output_path, "PNG")
        print("SUCCESS")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asset Station AI Background Remover")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to output image")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--model", type=str, default="u2net.onnx", help="Model filename to use")
    parser.add_argument("--edge-shift", type=int, default=0, help="Erode (<0) or Dilate (>0) mask")
    parser.add_argument("--edge-feather", type=int, default=0, help="Gaussian blur feather amount")

    args = parser.parse_args()

    remove_bg(args.input, args.output, args.model, args.gpu, args.edge_shift, args.edge_feather)
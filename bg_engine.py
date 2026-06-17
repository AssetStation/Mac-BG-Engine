import sys
import os
import argparse
import platform
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2 # NEW: For Edge Smoothing

def remove_bg(input_path, output_path, model_name="u2net.onnx", use_gpu=False, smooth_edges=True):
    try:
        # 1. Hardware Providers
        if use_gpu:
            if platform.system() == "Darwin":
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # 2. Bulletproof Path Resolution
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
        model_path = os.path.join(base_dir, model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find AI model at: {model_path}")

        # 3. Speed Optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        
        # 4. SMART RESOLUTION ROUTING
        # isnet-anime requires 1024x1024 for sharp hair. u2net uses 320x320.
        is_isnet = "isnet" in model_name.lower()
        target_size = 1024 if is_isnet else 320

        img = Image.open(input_path).convert("RGB")
        original_size = img.size
        img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # 5. SMART NORMALIZATION
        img_data = np.array(img_resized).astype(np.float32) / 255.0
        
        if is_isnet:
            # IS-Net math
            img_data = (img_data - [0.5, 0.5, 0.5]) / [1.0, 1.0, 1.0]
        else:
            # U2Net ImageNet math
            img_data = (img_data - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
        
        # 6. Process the image
        inputs = {session.get_inputs()[0].name: img_data}
        ort_outs = session.run(None, inputs)
        
        # 7. Extract Mask
        mask = ort_outs[0][0, 0, :, :]
        
        # Max-Min Normalization to ensure full opacity range
        mask_min = np.min(mask)
        mask_max = np.max(mask)
        if mask_max > mask_min:
            mask = (mask - mask_min) / (mask_max - mask_min)
            
        mask = (mask * 255).astype(np.uint8)

        # 8. IMPROVED ALPHA MATTING (Edge Smoothing)
        if smooth_edges:
            # 1. Binarize to remove gray ghosting/halos
            _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
            
            # 2. Erode the mask slightly to trim background edge bleed
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            # 3. Apply Gaussian blur to create a smooth, natural anti-aliased edge
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Resize the clean mask back to original resolution
        final_mask = Image.fromarray(mask).resize(original_size, Image.Resampling.LANCZOS)
        
        # Apply and save
        img.putalpha(final_mask)
        img.save(output_path, "PNG")
        print("SUCCESS")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asset Station AI Background Remover")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to output image")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    # 🟢 NEW: Allow JavaScript to pick the model!
    parser.add_argument("--model", type=str, default="u2net.onnx", help="Model filename to use")
    
    args = parser.parse_args()
    
    remove_bg(args.input, args.output, args.model, args.gpu)
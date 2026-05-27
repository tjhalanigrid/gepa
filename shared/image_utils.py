#!/usr/bin/env python3
"""
Shared Image Utility Helpers
Handles Base64 encoding, resizing, JPEG compression, and VLM payload optimization.
"""

import os

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

def get_optimized_image_paths(image_paths: list) -> list:
    """
    VLM payload optimization:
    Converts huge PNGs (e.g., 14MB) to lightweight, compressed JPEGs (e.g., 200KB)
    and resizes them if they exceed 1280px in any dimension.
    This prevents Ollama context/GPU OOM failures and speeds up inference by 10x.
    """
    if not HAS_PILLOW:
        print("[Warning] Pillow (PIL) is not installed. Sending original uncompressed images to VLM.")
        return image_paths

    # Save to a centralized temporary optimized folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(current_dir, "../models/vlm_reasoning/temp_optimized")
    os.makedirs(temp_dir, exist_ok=True)
    
    optimized_paths = []
    for path in image_paths:
        try:
            if not os.path.exists(path):
                print(f"[Warning] Image path not found: {path}")
                optimized_paths.append(path)
                continue
                
            filename = os.path.basename(path)
            name_part, _ = os.path.splitext(filename)
            opt_path = os.path.join(temp_dir, f"{name_part}_opt.jpg")
            
            # If already processed, reuse it to save time
            if os.path.exists(opt_path):
                optimized_paths.append(opt_path)
                continue
                
            with Image.open(path) as img:
                # Convert to RGB (in case of RGBA/PNG transparency)
                rgb_img = img.convert("RGB")
                
                # Resize if extremely large (e.g., > 1280px)
                max_dim = 1280
                w, h = rgb_img.size
                if w > max_dim or h > max_dim:
                    if w > h:
                        new_w = max_dim
                        new_h = int(h * (max_dim / w))
                    else:
                        new_h = max_dim
                        new_w = int(w * (max_dim / h))
                    rgb_img = rgb_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Compress as JPEG (quality 85 keeps damage details highly visible)
                rgb_img.save(opt_path, "JPEG", quality=85)
                optimized_paths.append(opt_path)
        except Exception as e:
            print(f"[Warning] Failed to optimize image {path}: {e}. Falling back to original.")
            optimized_paths.append(path)
            
    return optimized_paths

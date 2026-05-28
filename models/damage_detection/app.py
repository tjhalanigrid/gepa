import os
import json
import cv2
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO, SAM

# ==========================================
# 1. THE AUTOMATED DOWNLOADER GUARDRAIL
# ==========================================
def ensure_weights(file_path, download_url=None):
    """
    Checks if a model weight file exists in the specified folder. 
    If it exists, it skips downloading. If missing, it downloads it.
    """
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # THE GUARDRAIL: If the file is already in the models folder, stop and use it.
    if os.path.exists(file_path):
        print(f"[Setup] ✅ Found existing weights at: {file_path}")
        return file_path

    # FALLBACK: Only triggers if the file is genuinely missing
    if download_url:
        print(f"[Setup] ⚠️ Missing '{file_path}'. Downloading now...")
        try:
            urllib.request.urlretrieve(download_url, file_path)
            print(f"[Setup] ✅ Download complete: {file_path}")
        except Exception as e:
            print(f"[Setup] ❌ CRITICAL ERROR: Failed to download weights. {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    else:
        raise FileNotFoundError(
            f"\n[Setup] ❌ ERROR: '{file_path}' is missing!\n"
            f"You need to manually place your custom YOLO model in this location before running the app."
        )
    return file_path

# ==========================================
# 2. PATH CONFIGURATIONS & INITIALIZATION
# ==========================================
# Fixed to explicitly look inside the models/ folder
YOLO_WEIGHTS_PATH = os.path.join("models", "best.pt")
SAM_WEIGHTS_PATH = os.path.join("models", "sam2.1_b.pt")
SAM_DOWNLOAD_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt"

INPUT_DIR = "examples"  
OUTPUT_DIR = "outputs"
VISUAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "visual_proofs")

os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)

print("--- BOOTING ML PIPELINE ---")
# The script will now check the models/ folder first. 
# Since your screenshot shows they are there, it will skip downloading entirely!
ensure_weights(SAM_WEIGHTS_PATH, SAM_DOWNLOAD_URL)
ensure_weights(YOLO_WEIGHTS_PATH, download_url=None) 

print("🚀 Loading AI Models into RAM...")
box_model = YOLO(YOLO_WEIGHTS_PATH)
sam_model = SAM(SAM_WEIGHTS_PATH)

# ==========================================
# 3. CORE PROCESSING LOGIC
# ==========================================
def process_example_batch(conf_threshold=0.15, overlap_threshold=0.30):
    master_json_output = {}
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    input_images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]
    
    if not input_images:
        print(f"⚠️ No images found in '{INPUT_DIR}' folder. Please add your test images.")
        return

    print(f"📦 Found {len(input_images)} images to process.")

    for img_name in input_images:
        img_path = os.path.join(INPUT_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        print(f"🔍 Analyzing: {img_name}...")
        
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # --- STEP A: YOLO Bounding Boxes ---
        box_results = box_model.predict(img_path, conf=conf_threshold, verbose=False)[0]
        boxes_xyxy = box_results.boxes.xyxy.tolist() if box_results.boxes else []
        class_ids = box_results.boxes.cls.tolist() if box_results.boxes else []
        confidences = box_results.boxes.conf.tolist() if box_results.boxes else []
        
        yolo_plotted_bgr = box_results.plot()
        yolo_plotted_rgb = cv2.cvtColor(yolo_plotted_bgr, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(yolo_plotted_rgb)
        ax.axis('off')
        plt.savefig(os.path.join(VISUAL_OUTPUT_DIR, f"{base_name}_boxes.jpg"), bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
        
        image_record = []
        final_masks_to_plot = []
        
        # --- STEP B: SAM 2 Masks & Cross-Referencing ---
        if boxes_xyxy:
            sam_results = sam_model.predict(img_path, bboxes=boxes_xyxy, verbose=False)[0]
            
            if sam_results.masks is not None:
                for mask_idx, mask_data in enumerate(sam_results.masks.xy):
                    m_x_min, m_y_min = mask_data[:, 0].min(), mask_data[:, 1].min()
                    m_x_max, m_y_max = mask_data[:, 0].max(), mask_data[:, 1].max()
                    
                    overlapping_classes = []
                    detected_scores = {}
                    
                    for box, cls_id, score in zip(boxes_xyxy, class_ids, confidences):
                        b_x_min, b_y_min, b_x_max, b_y_max = box
                        class_name = box_model.names[int(cls_id)]
                        
                        inter_x_min = max(m_x_min, b_x_min)
                        inter_y_min = max(m_y_min, b_y_min)
                        inter_x_max = min(m_x_max, b_x_max)
                        inter_y_max = min(m_y_max, b_y_max)
                        
                        if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                            mask_area = (m_x_max - m_x_min) * (m_y_max - m_y_min)
                            
                            if (inter_area / mask_area) > overlap_threshold:
                                overlapping_classes.append(class_name)
                                detected_scores[class_name] = round(float(score), 2)
                    
                    unique_classes = list(set(overlapping_classes)) or ["unknown_damage"]
                    
                    image_record.append({
                        "damage_id": f"region_{mask_idx + 1}",
                        "bounding_box_xyxy": [round(x, 2) for x in [float(m_x_min), float(m_y_min), float(m_x_max), float(m_y_max)]],
                        "mask_polygon_coordinates": mask_data.tolist(),
                        "detected_types": unique_classes,
                        "confidence_scores": detected_scores
                    })
                    
                    final_masks_to_plot.append({"polygon": mask_data, "classes": unique_classes})
        
        master_json_output[img_name] = image_record
        
        indiv_json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
        with open(indiv_json_path, "w") as f:
            json.dump(image_record, f, indent=4)
        
        # --- STEP C: Save Stand-Alone SAM Mask Image ---
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_rgb)
        
        for item in final_masks_to_plot:
            poly = item["polygon"]
            labels = ", ".join(item["classes"])
            ax.plot(poly[:, 0], poly[:, 1], color='#00FF00', linewidth=2)
            ax.fill(poly[:, 0], poly[:, 1], color='#00FF00', alpha=0.4)
            ax.text(poly[:, 0].min(), poly[:, 1].min() - 8, labels, 
                    color='white', fontsize=10, fontweight='bold', 
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1))
        
        ax.axis('off')
        plt.savefig(os.path.join(VISUAL_OUTPUT_DIR, f"{base_name}_masks.jpg"), bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
        
    # --- STEP D: Save Consolidated Master JSON ---
    master_json_path = os.path.join(OUTPUT_DIR, "master_damage_report.json")
    with open(master_json_path, "w") as f:
        json.dump(master_json_output, f, indent=4)
        
    print(f"\n🎉 Examples batch processing finished successfully!")
    print(f"📁 Images with boxes/masks exported to: {VISUAL_OUTPUT_DIR}")
    print(f"📄 Target JSON outputs built inside: {OUTPUT_DIR}/")

if __name__ == "__main__":
    process_example_batch(conf_threshold=0.15)
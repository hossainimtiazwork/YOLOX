import json
import cv2
import os
import numpy as np
import argparse

def visualize_polygons(json_path, image_dir, output_dir, num_images=5):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    images = data['images']
    # Use images that have annotations
    image_ids_with_annos = set(anno['image_id'] for anno in data['annotations'] if 'polygon' in anno)
    valid_images = [img for img in images if img['id'] in image_ids_with_annos]
    
    selected_images = valid_images[:num_images]
    
    for img_info in selected_images:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(image_dir, file_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found.")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Failed to load image {img_path}")
            continue
            
        annos = [anno for anno in data['annotations'] if anno['image_id'] == img_id]
        
        for anno in annos:
            if 'polygon' in anno:
                poly = np.array(anno['polygon'], dtype=np.int32).reshape((-1, 2))
                cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Draw label
                x, y = poly[0]
                category_name = "cheetah" if anno['category_id'] == 1 else "human"
                cv2.putText(img, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = os.path.join(output_dir, f"vis_{file_name}")
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COCO polygon annotations")
    parser.add_argument("--json", required=True, help="Path to COCO JSON file")
    parser.add_argument("--img_dir", required=True, help="Directory containing images")
    parser.add_argument("--out_dir", default="YOLOX_outputs/visualization", help="Directory to save visualizations")
    parser.add_argument("--num", type=int, default=5, help="Number of images to visualize")
    args = parser.parse_args()
    
    visualize_polygons(args.json, args.img_dir, args.out_dir, args.num)

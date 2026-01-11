import json
import argparse
import os

def convert_coco_to_poly(json_path, save_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for anno in data['annotations']:
        if 'bbox' in anno and 'polygon' not in anno:
            x, y, w, h = anno['bbox']
            # Convert [x, y, w, h] to [x1, y1, x2, y2, x3, y3, x4, y4]
            # (x1,y1) -> top-left
            # (x2,y1) -> top-right
            # (x2,y2) -> bottom-right
            # (x1,y2) -> bottom-left
            anno['polygon'] = [x, y, x + w, y, x + w, y + h, x, y + h]
            
    with open(save_path, 'w') as f:
        json.dump(data, f)
    print(f"Successfully converted {json_path} to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO bbox to polygon format")
    parser.add_argument("json_file", help="Path to COCO JSON file")
    parser.add_argument("--save_file", help="Path to save converted JSON file")
    args = parser.parse_args()
    
    save_file = args.save_file if args.save_file else args.json_file.replace(".json", "_poly.json")
    convert_coco_to_poly(args.json_file, save_file)

"""
MTdata_makelabel.py - Multi-Annotation Generation Tool
Automatic generation of Caption, Depth, and Pose annotations for image datasets
"""

import os
import json
import glob
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from collections import defaultdict
import argparse

from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from transformers import pipeline as hf_pipeline
from mmpose.apis import inference_bottom_up_pose_model, init_pose_model


CONFIG = {
    "models": {
        "blip2": "/path/to/blip2",
        "clip": "/path/to/clip-vit-large-patch14",
        "depth": "/path/to/depth-anything",
        "mmpose_config": "/path/to/mmpose/config.py",
        "mmpose_checkpoint": "/path/to/mmpose/checkpoint.pth"
    },
    
    "datasets": {
        "Dataset1": {
            "image_dir": "/path/to/dataset1/images",
            "output_dir": "/path/to/dataset1/output"
        },
        "Dataset2": {
            "image_dir": "/path/to/dataset2/images",
            "output_dir": "/path/to/dataset2/output"
        }
    },
    
    "caption": {
        "num_captions": 5,
        "max_new_tokens": 50,
        "top_p": 0.9,
        "temperature": 0.7
    },
    
    "pose": {
        "max_person_num": 20,
        "keypoint_num": 17,
        "keypoint_thresh": 0.02,
        "pose_nms_thr": 1.0
    }
}


class CaptionGenerator:
    
    def __init__(self, device="cuda"):
        self.device = device
        print(f"Loading Caption models to {device}...")
        
        self.blip_processor = Blip2Processor.from_pretrained(CONFIG["models"]["blip2"])
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            CONFIG["models"]["blip2"],
            torch_dtype=torch.float16
        )
        self.blip_model.to(device)
        
        self.clip_processor = CLIPProcessor.from_pretrained(CONFIG["models"]["clip"])
        self.clip_model = CLIPModel.from_pretrained(CONFIG["models"]["clip"])
        self.clip_model.to(device)
        
        print("Caption models loaded successfully")
    
    def generate_captions(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            captions = []
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device, torch.float16)
            
            for _ in range(CONFIG["caption"]["num_captions"]):
                generated_ids = self.blip_model.generate(
                    **inputs,
                    max_new_tokens=CONFIG["caption"]["max_new_tokens"],
                    do_sample=True,
                    top_p=CONFIG["caption"]["top_p"],
                    temperature=CONFIG["caption"]["temperature"]
                )
                generated_text = self.blip_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
                captions.append(generated_text)
            
            scores = []
            for caption in captions:
                inputs = self.clip_processor(
                    text=caption, images=image, 
                    return_tensors="pt", padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    score = outputs.logits_per_image.item()
                scores.append(score)
            
            best_idx = scores.index(max(scores))
            return captions[best_idx], max(scores)
            
        except Exception as e:
            print(f"Caption generation failed for {image_path}: {e}")
            return None, None


class DepthGenerator:
    
    def __init__(self, device="cuda:0"):
        self.device = device
        print(f"Loading Depth model to {device}...")
        
        self.pipe = hf_pipeline(
            "depth-estimation",
            model=CONFIG["models"]["depth"],
            device=device
        )
        
        print("Depth model loaded successfully")
    
    def generate_depth(self, image_path, output_path):
        try:
            if os.path.exists(output_path):
                return True, "exists"
            
            image = Image.open(image_path)
            predictions = self.pipe(image)
            depth_image = predictions["depth"]
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            depth_image.save(output_path)
            return True, "success"
            
        except Exception as e:
            return False, str(e)


class PoseGenerator:
    
    def __init__(self, device="cuda"):
        self.device = device
        print(f"Loading Pose model to {device}...")
        
        self.model = init_pose_model(
            CONFIG["models"]["mmpose_config"],
            CONFIG["models"]["mmpose_checkpoint"],
            device=device
        )
        
        self.max_person_num = CONFIG["pose"]["max_person_num"]
        self.keypoint_num = CONFIG["pose"]["keypoint_num"]
        self.keypoint_thresh = CONFIG["pose"]["keypoint_thresh"]
        
        print("Pose model loaded successfully")
    
    def generate_pose(self, image_path, output_path):
        try:
            if os.path.exists(output_path):
                return True, "exists"
            
            image = cv2.imread(image_path)
            if image is None:
                return False, "failed to read image"
            
            mmpose_results = inference_bottom_up_pose_model(
                self.model,
                image,
                dataset="BottomUpCocoDataset",
                dataset_info=None,
                pose_nms_thr=CONFIG["pose"]["pose_nms_thr"],
                return_heatmap=False,
                outputs=None,
            )[0]
            
            pose_array = np.array(mmpose_results)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path, pose_array)
            
            return True, "success"
            
        except Exception as e:
            return False, str(e)


class DatasetAnnotator:
    
    def __init__(self, dataset_name, enable_caption=True, enable_depth=True, enable_pose=True, gpu_id=0):
        self.dataset_name = dataset_name
        self.dataset_config = CONFIG["datasets"][dataset_name]
        self.enable_caption = enable_caption
        self.enable_depth = enable_depth
        self.enable_pose = enable_pose
        
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        if enable_caption:
            self.caption_gen = CaptionGenerator(device=device)
        if enable_depth:
            self.depth_gen = DepthGenerator(device=device)
        if enable_pose:
            self.pose_gen = PoseGenerator(device=device)
        
        self.output_dir = self.dataset_config["output_dir"]
        if enable_caption:
            self.caption_dir = os.path.join(self.output_dir, "captions")
            os.makedirs(self.caption_dir, exist_ok=True)
        if enable_depth:
            self.depth_dir = os.path.join(self.output_dir, "depth")
            os.makedirs(self.depth_dir, exist_ok=True)
        if enable_pose:
            self.pose_dir = os.path.join(self.output_dir, "pose")
            os.makedirs(self.pose_dir, exist_ok=True)
    
    def get_image_list(self):
        image_dir = self.dataset_config["image_dir"]
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(image_dir, "**", ext)
            image_files.extend(glob.glob(pattern, recursive=True))
        
        return sorted(image_files)
    
    def process_single_image(self, image_path):
        results = {
            "image_path": image_path,
            "caption": None,
            "depth": None,
            "pose": None
        }
        
        image_dir = self.dataset_config["image_dir"]
        rel_path = os.path.relpath(image_path, image_dir)
        filename = os.path.basename(image_path)
        filename_noext = os.path.splitext(filename)[0]
        
        if self.enable_caption:
            try:
                caption, score = self.caption_gen.generate_captions(image_path)
                results["caption"] = {"text": caption, "score": score}
            except Exception as e:
                results["caption"] = {"error": str(e)}
        
        if self.enable_depth:
            depth_rel_dir = os.path.dirname(rel_path)
            depth_output_dir = os.path.join(self.depth_dir, depth_rel_dir)
            depth_output_path = os.path.join(depth_output_dir, filename)
            
            success, msg = self.depth_gen.generate_depth(image_path, depth_output_path)
            results["depth"] = {"success": success, "message": msg, "path": depth_output_path}
        
        if self.enable_pose:
            pose_rel_dir = os.path.dirname(rel_path)
            pose_output_dir = os.path.join(self.pose_dir, pose_rel_dir)
            pose_output_path = os.path.join(pose_output_dir, filename_noext + ".npz")
            
            success, msg = self.pose_gen.generate_pose(image_path, pose_output_path)
            results["pose"] = {"success": success, "message": msg, "path": pose_output_path}
        
        return results
    
    def process_dataset(self):
        print(f"\n{'='*80}")
        print(f"Processing dataset: {self.dataset_name}")
        print(f"{'='*80}")
        print(f"Image directory: {self.dataset_config['image_dir']}")
        print(f"Output directory: {self.dataset_config['output_dir']}")
        print(f"Enabled: Caption={self.enable_caption}, Depth={self.enable_depth}, Pose={self.enable_pose}")
        
        image_list = self.get_image_list()
        print(f"Found {len(image_list)} images")
        
        if len(image_list) == 0:
            print("Warning: No images found!")
            return
        
        all_results = []
        caption_data = []
        
        for image_path in tqdm(image_list, desc=f"Processing {self.dataset_name}"):
            result = self.process_single_image(image_path)
            all_results.append(result)
            
            if self.enable_caption and result["caption"] and "text" in result["caption"]:
                rel_path = os.path.relpath(image_path, self.dataset_config["image_dir"])
                caption_data.append({
                    "image": rel_path,
                    "caption": result["caption"]["text"],
                    "score": result["caption"]["score"]
                })
        
        if self.enable_caption and caption_data:
            caption_json_path = os.path.join(self.caption_dir, f"{self.dataset_name}_captions.json")
            with open(caption_json_path, 'w', encoding='utf-8') as f:
                json.dump(caption_data, f, ensure_ascii=False, indent=2)
            
            caption_txt_path = os.path.join(self.caption_dir, f"{self.dataset_name}_captions.txt")
            with open(caption_txt_path, 'w', encoding='utf-8') as f:
                for item in caption_data:
                    f.write(f"{item['image']}\t{item['caption']}\t{item['score']:.4f}\n")
            
            print(f"Captions saved to: {caption_json_path}")
        
        self.print_statistics(all_results)
        return all_results
    
    def print_statistics(self, results):
        print(f"\n{'='*80}")
        print(f"Dataset {self.dataset_name} completed")
        print(f"{'='*80}")
        
        total = len(results)
        print(f"Total images: {total}")
        
        if self.enable_caption:
            caption_success = sum(1 for r in results if r["caption"] and "text" in r["caption"])
            print(f"Caption success: {caption_success}/{total}")
        
        if self.enable_depth:
            depth_success = sum(1 for r in results if r["depth"] and r["depth"]["success"])
            depth_exist = sum(1 for r in results if r["depth"] and r["depth"]["message"] == "exists")
            print(f"Depth success: {depth_success}/{total} (existed: {depth_exist})")
        
        if self.enable_pose:
            pose_success = sum(1 for r in results if r["pose"] and r["pose"]["success"])
            pose_exist = sum(1 for r in results if r["pose"] and r["pose"]["message"] == "exists")
            print(f"Pose success: {pose_success}/{total} (existed: {pose_exist})")


def process_single_dataset_worker(args):
    dataset_name, enable_caption, enable_depth, enable_pose, gpu_id = args
    
    try:
        annotator = DatasetAnnotator(
            dataset_name=dataset_name,
            enable_caption=enable_caption,
            enable_depth=enable_depth,
            enable_pose=enable_pose,
            gpu_id=gpu_id
        )
        
        results = annotator.process_dataset()
        return dataset_name, True, results
        
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        return dataset_name, False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Multi-Dataset Annotation Generation Tool')
    
    parser.add_argument('--datasets', type=str, default='all',
                      help='Datasets to process (comma-separated or "all")')
    parser.add_argument('--enable-caption', action='store_true', default=False,
                      help='Enable Caption generation')
    parser.add_argument('--enable-depth', action='store_true', default=False,
                      help='Enable Depth generation')
    parser.add_argument('--enable-pose', action='store_true', default=False,
                      help='Enable Pose generation')
    parser.add_argument('--enable-all', action='store_true', default=False,
                      help='Enable all annotation types')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU ID to use')
    parser.add_argument('--multi-gpu', action='store_true',
                      help='Use multiple GPUs for parallel processing')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                      help='GPU IDs for multi-GPU mode (comma-separated)')
    
    args = parser.parse_args()
    
    if args.enable_all:
        enable_caption = True
        enable_depth = True
        enable_pose = True
    else:
        enable_caption = args.enable_caption
        enable_depth = args.enable_depth
        enable_pose = args.enable_pose
    
    if not (enable_caption or enable_depth or enable_pose):
        print("Error: At least one annotation type must be enabled")
        return
    
    if args.datasets.lower() == 'all':
        dataset_names = list(CONFIG["datasets"].keys())
    else:
        dataset_names = [d.strip() for d in args.datasets.split(',')]
        for name in dataset_names:
            if name not in CONFIG["datasets"]:
                print(f"Error: Unknown dataset '{name}'")
                print(f"Available datasets: {', '.join(CONFIG['datasets'].keys())}")
                return
    
    print(f"{'='*80}")
    print(f"Multi-Dataset Annotation Generation System")
    print(f"{'='*80}")
    print(f"Datasets to process: {', '.join(dataset_names)}")
    print(f"Enabled: Caption={enable_caption}, Depth={enable_depth}, Pose={enable_pose}")
    
    if args.multi_gpu and len(dataset_names) > 1:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
        print(f"Using multi-GPU mode: {gpu_ids}")
        
        tasks = []
        for idx, dataset_name in enumerate(dataset_names):
            gpu_id = gpu_ids[idx % len(gpu_ids)]
            tasks.append((dataset_name, enable_caption, enable_depth, enable_pose, gpu_id))
        
        print(f"Starting {len(tasks)} processes...")
        with mp.Pool(min(len(tasks), len(gpu_ids))) as pool:
            results = pool.map(process_single_dataset_worker, tasks)
        
        print(f"\n{'='*80}")
        print("All datasets completed")
        print(f"{'='*80}")
        for dataset_name, success, _ in results:
            status = "Success" if success else "Failed"
            print(f"{dataset_name}: {status}")
    else:
        for dataset_name in dataset_names:
            annotator = DatasetAnnotator(
                dataset_name=dataset_name,
                enable_caption=enable_caption,
                enable_depth=enable_depth,
                enable_pose=enable_pose,
                gpu_id=args.gpu
            )
            
            annotator.process_dataset()
    
    print(f"\n{'='*80}")
    print("All tasks completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

import os
import json
import csv
import shutil
from tqdm import tqdm
import numpy as np
from collections import defaultdict

DATASETS_CONFIG = {
    "Dataset1": "/path/to/dataset1/pose",
    "Dataset2": "/path/to/dataset2/pose"
}

CLIP_CSV_CONFIG = {
    "Dataset1": "/path/to/dataset1/clip_scores.csv",
    "Dataset2": "/path/to/dataset2/clip_scores.csv"
}

FILTER_CRITERIA = {
    "density": {
        "min": 0.3,
        "max": 1.0
    },
    "complexity": {
        "min": 0.4,
        "max": 1.0
    },
    "clip_score": {
        "min": 0.5,
        "max": 1.0
    }
}

class DatasetFilter:
    
    def __init__(self, analysis_json_path, clip_csv_path=None, dataset_name="Unknown"):
        self.dataset_name = dataset_name
        self.analysis_json_path = analysis_json_path
        self.clip_csv_path = clip_csv_path
        self.analysis_data = self._load_analysis_data()
        self.clip_scores = self._load_clip_scores() if clip_csv_path else {}
        self.filtered_results = []
    
    def _load_analysis_data(self):
        try:
            with open(self.analysis_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ 成功加载分析数据: {self.analysis_json_path}")
            print(f"  总图片数: {data.get('total_images', 0)}")
            return data
        except Exception as e:
            print(f"✗ 加载分析数据失败: {e}")
            return None
    
    def _load_clip_scores(self):
        try:
            clip_scores = {}
            with open(self.clip_csv_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                
                for idx, row in enumerate(csv_reader):
                    if row and row[0].strip():
                        try:
                            score = float(row[0])
                            clip_scores[idx] = score
                        except ValueError:
                            continue
            
            print(f"✓ 成功加载CLIP分数: {len(clip_scores)} 个")
            return clip_scores
        except Exception as e:
            print(f"✗ 加载CLIP分数失败: {e}")
            return {}
    
    def filter_data(self, criteria=None):
        if criteria is None:
            criteria = FILTER_CRITERIA
        
        if not self.analysis_data or 'detailed_results' not in self.analysis_data:
            print("没有有效的分析数据可以筛选")
            return []
        
        detailed_results = self.analysis_data['detailed_results']
        total_count = len(detailed_results)
        
        print(f"\n{'='*60}")
        print(f"开始筛选数据集: {self.dataset_name}")
        print(f"{'='*60}")
        print(f"总图片数: {total_count}")
        print(f"\n筛选条件:")
        print(f"  密集程度: {criteria['density']['min']:.2f} - {criteria['density']['max']:.2f}")
        print(f"  姿态复杂度: {criteria['complexity']['min']:.2f} - {criteria['complexity']['max']:.2f}")
        print(f"  CLIP分数: {criteria['clip_score']['min']:.2f} - {criteria['clip_score']['max']:.2f}")
        
        filtered_results = []
        
        for idx, result in enumerate(tqdm(detailed_results, desc="Filtering")):
            density_score = result.get('density_score', 0)
            if not (criteria['density']['min'] <= density_score <= criteria['density']['max']):
                continue
            
            complexity_score = result.get('avg_complexity_score', 0)
            if not (criteria['complexity']['min'] <= complexity_score <= criteria['complexity']['max']):
                continue
            
            if self.clip_scores:
                clip_score = self.clip_scores.get(idx, 0)
                if not (criteria['clip_score']['min'] <= clip_score <= criteria['clip_score']['max']):
                    continue
                result['clip_score'] = clip_score
            else:
                result['clip_score'] = None
            
            filtered_results.append(result)
        
        self.filtered_results = filtered_results
        
        filtered_count = len(filtered_results)
        print(f"\nFiltering results:")
        print(f"  Before: {total_count} images")
        print(f"  After: {filtered_count} images")
        print(f"  Filter rate: {filtered_count/total_count*100:.2f}%")
        
        return filtered_results
    
    def export_filtered_list(self, output_path):
        if not self.filtered_results:
            print("没有筛选结果可以导出")
            return
        
        try:
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                # 写入标题
                headers = ['文件路径', '人数', '密集程度', '姿态复杂度', 'CLIP分数']
                writer.writerow(headers)
                
                # 写入数据
                for result in self.filtered_results:
                    row = [
                        result['file_path'],
                        result['num_persons'],
                        f"{result['density_score']:.4f}",
                        f"{result['avg_complexity_score']:.4f}",
                        f"{result['clip_score']:.4f}" if result['clip_score'] is not None else "N/A"
                    ]
                    writer.writerow(row)
            
            print(f"✓ 筛选列表已导出到: {output_path}")
        
        except Exception as e:
            print(f"✗ 导出筛选列表失败: {e}")
    
    def copy_filtered_images(self, output_dir, copy_pose=True):
        if not self.filtered_results:
            print("No filtered results to copy")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        images_dir = os.path.join(output_dir, "images")
        poses_dir = os.path.join(output_dir, "pose")
        os.makedirs(images_dir, exist_ok=True)
        if copy_pose:
            os.makedirs(poses_dir, exist_ok=True)
        
        print(f"\nCopying files to: {output_dir}")
        
        success_count = 0
        failed_count = 0
        
        for result in tqdm(self.filtered_results, desc="Copying files"):
            file_path = result['file_path']
            pose_file = os.path.join(os.path.dirname(self.analysis_json_path), file_path)
            image_file = pose_file.replace('/pose', '').replace('.npz', '.jpg').replace('.npy', '.jpg')
            
            try:
                if os.path.exists(image_file):
                    dest_image = os.path.join(images_dir, os.path.basename(image_file))
                    shutil.copy2(image_file, dest_image)
                else:
                    print(f"Warning: Image not found {image_file}")
                    failed_count += 1
                    continue
                
                if copy_pose and os.path.exists(pose_file):
                    dest_pose = os.path.join(poses_dir, os.path.basename(pose_file))
                    shutil.copy2(pose_file, dest_pose)
                
                success_count += 1
                
            except Exception as e:
                print(f"Copy failed {file_path}: {e}")
                failed_count += 1
        
        print(f"\nCopy completed:")
        print(f"  Success: {success_count} files")
        print(f"  Failed: {failed_count} files")
    
    def print_statistics(self):
        if not self.filtered_results:
            print("没有筛选结果")
            return
        
        print(f"\n{'='*60}")
        print(f"筛选结果统计 - {self.dataset_name}")
        print(f"{'='*60}")
        
        # 提取各项指标
        person_counts = [r['num_persons'] for r in self.filtered_results]
        density_scores = [r['density_score'] for r in self.filtered_results]
        complexity_scores = [r['avg_complexity_score'] for r in self.filtered_results]
        clip_scores = [r['clip_score'] for r in self.filtered_results if r['clip_score'] is not None]
        
        print(f"\n人数统计:")
        print(f"  平均人数: {np.mean(person_counts):.2f}")
        print(f"  人数范围: {min(person_counts)} - {max(person_counts)}")
        
        print(f"\n密集程度统计:")
        print(f"  平均值: {np.mean(density_scores):.4f}")
        print(f"  范围: {min(density_scores):.4f} - {max(density_scores):.4f}")
        
        print(f"\n姿态复杂度统计:")
        print(f"  平均值: {np.mean(complexity_scores):.4f}")
        print(f"  范围: {min(complexity_scores):.4f} - {max(complexity_scores):.4f}")
        
        if clip_scores:
            print(f"\nCLIP分数统计:")
            print(f"  平均值: {np.mean(clip_scores):.4f}")
            print(f"  范围: {min(clip_scores):.4f} - {max(clip_scores):.4f}")


def process_all_datasets(output_base_dir="./filtered_data"):
    print("="*80)
    print("Multi-Dataset Filtering System")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    all_filtered_results = []
    dataset_filters = []
    
    for dataset_name in DATASETS_CONFIG.keys():
        print(f"\n\n{'#'*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'#'*80}")
        
        analysis_json = f"{dataset_name}_pose_analysis_results.json"
        
        if not os.path.exists(analysis_json):
            print(f"Warning: Analysis results file not found {analysis_json}")
            continue
        
        clip_csv = CLIP_CSV_CONFIG.get(dataset_name)
        if clip_csv and not os.path.exists(clip_csv):
            print(f"Warning: CLIP score file not found {clip_csv}, skipping CLIP filtering")
            clip_csv = None
        
        filter_obj = DatasetFilter(
            analysis_json_path=analysis_json,
            clip_csv_path=clip_csv,
            dataset_name=dataset_name
        )
        
        filtered_results = filter_obj.filter_data(FILTER_CRITERIA)
        
        if filtered_results:
            dataset_filters.append(filter_obj)
            all_filtered_results.extend(filtered_results)
            
            filter_obj.print_statistics()
            
            output_csv = os.path.join(output_base_dir, f"{dataset_name}_filtered_list.csv")
            filter_obj.export_filtered_list(output_csv)
            
            output_dataset_dir = os.path.join(output_base_dir, dataset_name)
            filter_obj.copy_filtered_images(output_dataset_dir, copy_pose=True)
    
    print(f"\n\n{'='*80}")
    print("Summary Report")
    print(f"{'='*80}")
    print(f"Datasets processed: {len(dataset_filters)}")
    print(f"Total filtered images: {len(all_filtered_results)}")
    
    if all_filtered_results:
        summary_csv = os.path.join(output_base_dir, "all_datasets_filtered_summary.csv")
        
        try:
            with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                headers = ['Dataset', 'File Path', 'Person Count', 'Density', 'Complexity', 'CLIP Score']
                writer.writerow(headers)
                
                for filter_obj in dataset_filters:
                    for result in filter_obj.filtered_results:
                        row = [
                            filter_obj.dataset_name,
                            result['file_path'],
                            result['num_persons'],
                            f"{result['density_score']:.4f}",
                            f"{result['avg_complexity_score']:.4f}",
                            f"{result['clip_score']:.4f}" if result['clip_score'] is not None else "N/A"
                        ]
                        writer.writerow(row)
            
            print(f"Summary report saved to: {summary_csv}")
        
        except Exception as e:
            print(f"Failed to generate summary report: {e}")


def custom_filter_example():
    print("="*80)
    print("Custom Filtering Example")
    print("="*80)
    
    dataset_name = list(DATASETS_CONFIG.keys())[0]
    analysis_json = f"{dataset_name}_pose_analysis_results.json"
    clip_csv = CLIP_CSV_CONFIG.get(dataset_name)
    
    if not os.path.exists(analysis_json):
        print(f"Error: Analysis results file not found {analysis_json}")
        return
    
    filter_obj = DatasetFilter(
        analysis_json_path=analysis_json,
        clip_csv_path=clip_csv,
        dataset_name=dataset_name
    )
    
    custom_criteria = {
        "density": {"min": 0.5, "max": 0.8},
        "complexity": {"min": 0.6, "max": 1.0},
        "clip_score": {"min": 0.7, "max": 1.0}
    }
    
    filtered_results = filter_obj.filter_data(custom_criteria)
    filter_obj.print_statistics()
    filter_obj.export_filtered_list(f"{dataset_name}_custom_filtered_list.csv")


def main():
    import sys
    
    mode = input("Select mode:\n1. Process all datasets (default)\n2. Custom filtering example\nEnter (1/2): ").strip()
    
    if mode == "2":
        custom_filter_example()
    else:
        output_dir = input("Enter output directory path (default: ./filtered_data): ").strip()
        if not output_dir:
            output_dir = "./filtered_data"
        
        process_all_datasets(output_base_dir=output_dir)
    
    print("\n" + "="*80)
    print("Processing completed!")
    print("="*80)


if __name__ == "__main__":
    main()

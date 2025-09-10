# -*- coding: utf-8 -*-
"""
VLM統合評価＆プロットスクリプト (大規模実行＆サマリーCSV追加版) 

本スクリプトの目的:
1. ★★★【新規】CLS, 平均特徴量について、カテゴリごとの要約CSVを最上位ディレクトリに出力する。★★★
2. 分析結果を「カテゴリ別ディレクトリ」内の「特徴量タイプ別CSV」に分割して出力する。
3. 4種類の画像特徴量について、全語彙との類似度Top-5を記録する。
4. 実験規模を各カテゴリ最大100サンプルに戻す。
"""
import os
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import random
import datetime
import traceback
from collections import Counter

# --- 1. 設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DATASET_NAME = 'coco'
DATA_DIR = "./data"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# --- ▼▼▼ ここで実験設定を切り替え ▼▼▼ ---
INPUT_IMAGE_MODE = "normal"
# ★★★変更点★★★ 実験規模を最大100に戻す
NUM_SAMPLES_PER_CATEGORY = 100 
CATEGORIES_TO_ANALYZE = ["cat", "dog", "car", "bicycle", "person", "apple", "chair", "book"]
# --- ▲▲▲ 設定はここまで ▲▲▲ ---

MASKED_IMAGE_DIR = os.path.join(DATA_DIR, "outline_masked_coco_val2017")
output_folder_name = f"projector_analysis_{INPUT_IMAGE_MODE}_split"
OUTPUT_DIR = os.path.join("evaluation_results", DATASET_NAME, output_folder_name, TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)
if INPUT_IMAGE_MODE == "masked":
    os.makedirs(MASKED_IMAGE_DIR, exist_ok=True)

# (CocoObjectDatasetクラスは変更なし)
class CocoObjectDataset(Dataset):
    def __init__(self, root, annFile_instances, categories, num_samples_per_cat, image_mode, masked_img_dir=None):
        self.root = root
        self.coco_instances = COCO(annFile_instances)
        self.image_mode = image_mode
        self.masked_img_dir = masked_img_dir
        self.data = []
        print(f"画像モード: '{self.image_mode}' でデータセットを準備しています...")
        print("指定されたカテゴリから画像とアノテーションをサンプリングしています...")
        for cat_name in categories:
            cat_ids = self.coco_instances.getCatIds(catNms=[cat_name]);
            if not cat_ids: continue
            img_ids_for_cat = self.coco_instances.getImgIds(catIds=cat_ids)
            sampled_img_ids = random.sample(img_ids_for_cat, min(num_samples_per_cat, len(img_ids_for_cat)))
            for img_id in sampled_img_ids:
                annIds = self.coco_instances.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
                if not annIds: continue
                anns = self.coco_instances.loadAnns(annIds)
                main_ann = max(anns, key=lambda x: x['area'])
                self.data.append({'img_id': img_id, 'label': cat_name, 'annotation': main_ann})
        if self.image_mode == "masked":
            print(f"マスク済み画像のキャッシュ先: {self.masked_img_dir}")

    def __getitem__(self, index):
        item = self.data[index]
        img_id, label, annotation = item['img_id'], item['label'], item['annotation']
        original_filename = self.coco_instances.loadImgs(img_id)[0]['file_name']
        original_image_path = os.path.join(self.root, original_filename)
        if self.image_mode == "normal":
            image = Image.open(original_image_path).convert('RGB')
        elif self.image_mode == "masked":
            masked_image_path = os.path.join(self.masked_img_dir, original_filename)
            if os.path.exists(masked_image_path):
                image = Image.open(masked_image_path).convert('RGB')
            else:
                from scipy.ndimage import binary_erosion
                original_image = Image.open(original_image_path).convert('RGB')
                mask = self.coco_instances.annToMask(annotation)
                eroded_mask = binary_erosion(mask, iterations=2)
                outline_mask = mask ^ eroded_mask
                image_np = np.array(original_image)
                processed_image_np = np.zeros_like(image_np)
                processed_image_np[eroded_mask] = image_np[eroded_mask]
                processed_image_np[outline_mask] = 255
                image = Image.fromarray(processed_image_np)
                image.save(masked_image_path)
        else:
            raise ValueError(f"無効なINPUT_IMAGE_MODEです: {self.image_mode}")
        return {'image': image, 'label': label, 'img_id': img_id}
        
    def __len__(self): return len(self.data)

# --- 2. ヘルパー関数 ---
def get_projector_features(image, model, processor):
    """射影層の出力のみを効率的に取得する関数"""
    image_tensor = processor.image_processor(image, return_tensors="pt")['pixel_values'].to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        vision_outputs = model.vision_tower(image_tensor, output_hidden_states=True)
        vision_last_hidden_state = vision_outputs.hidden_states[-1]
        
        projected_cls_feat = model.multi_modal_projector(vision_last_hidden_state[:, 0:1, :]).squeeze(0)
        projected_patch_feats = model.multi_modal_projector(vision_last_hidden_state[:, 1:, :]).squeeze(0)
        
        mean_no_cls = projected_patch_feats.mean(dim=0)
        all_feats = torch.cat([projected_cls_feat, projected_patch_feats], dim=0)
        mean_with_cls = all_feats.mean(dim=0)

    return projected_cls_feat.squeeze(0), mean_with_cls, mean_no_cls, projected_patch_feats

# --- 3. ★★★新規★★★ サマリー作成用の関数 ---
def create_and_save_summaries(results, output_dir):
    """グローバル特徴量の分析結果からサマリーを作成して保存する"""
    print("\nCreating summary files for global features...")
    global_feature_types = ["cls", "mean_with_cls", "mean_no_cls"]
    
    for f_type in global_feature_types:
        summary_rows = []
        # カテゴリごとにループ
        for category, feature_data in results.items():
            # 当該カテゴリ・特徴量タイプのデータが存在するか確認
            if not feature_data[f_type]:
                continue
            
            # 全サンプルにおけるTop-1単語をリストアップ
            top1_words = [row['top_1_word'] for row in feature_data[f_type]]
            
            if top1_words:
                # 最も頻繁に出現したTop-1単語とその回数を取得
                most_common_word, count = Counter(top1_words).most_common(1)[0]
                # 出現率を計算
                frequency = count / len(top1_words)
                
                summary_rows.append({
                    "category": category,
                    "most_common_top1_word": most_common_word,
                    "frequency": frequency,
                    "count": count,
                    "total_samples": len(top1_words)
                })
        
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            summary_csv_path = os.path.join(output_dir, f"{f_type}_summary.csv")
            df_summary.to_csv(summary_csv_path, index=False)
            print(f" -> Saved summary: {summary_csv_path}")

# --- 4. メイン分析関数 ---
def run_combined_analysis(model, processor, dataset, output_dir):
    print("-" * 50); print(f"🚀 Running Combined Top-5 Analysis...")
    tokenizer = processor.tokenizer
    
    print("Preparing the full vocabulary embedding matrix...")
    full_embed_matrix = model.get_input_embeddings().weight
    full_embed_matrix_norm = F.normalize(full_embed_matrix, p=2, dim=-1).to(DEVICE, dtype=torch.float16)
    vocab_size = tokenizer.vocab_size
    print(f"Full vocabulary size: {vocab_size} tokens.")
    
    results = {cat: {"cls": [], "mean_with_cls": [], "mean_no_cls": [], "patch": []} for cat in CATEGORIES_TO_ANALYZE}
    
    for sample in tqdm(dataset, desc="Analyzing Images"):
        try:
            image, label, img_id = sample['image'], sample['label'], sample['img_id']
            cls_feat, mean_with_cls, mean_no_cls, patch_feats = get_projector_features(image, model, processor)

            global_features = {
                "cls": cls_feat, "mean_with_cls": mean_with_cls, "mean_no_cls": mean_no_cls,
            }
            for f_type, f_vec in global_features.items():
                f_vec_norm = F.normalize(f_vec.unsqueeze(0), p=2, dim=-1)
                similarities = torch.matmul(f_vec_norm, full_embed_matrix_norm.T)
                top5_scores, top5_indices = torch.topk(similarities, k=5, dim=1)
                
                row = {"image_id": img_id}
                for k in range(5):
                    token_id = top5_indices[0, k].item()
                    row[f"top_{k+1}_token_id"] = token_id
                    row[f"top_{k+1}_word"] = tokenizer.decode([token_id])
                    row[f"top_{k+1}_score"] = top5_scores[0, k].item()
                results[label][f_type].append(row)

            patch_feats_norm = F.normalize(patch_feats, p=2, dim=-1)
            sim_matrix = torch.matmul(patch_feats_norm, full_embed_matrix_norm.T)
            top5_scores_patches, top5_indices_patches = torch.topk(sim_matrix, k=5, dim=1)

            for i in range(patch_feats.shape[0]):
                row = {"image_id": img_id, "patch_id": i}
                for k in range(5):
                    token_id = top5_indices_patches[i, k].item()
                    row[f"top_{k+1}_token_id"] = token_id
                    row[f"top_{k+1}_word"] = tokenizer.decode([token_id])
                    row[f"top_{k+1}_score"] = top5_scores_patches[i, k].item()
                results[label]["patch"].append(row)

        except Exception as e:
            print(f"\nError processing image {img_id}: {e}")
            traceback.print_exc()

    # --- ★★★変更点★★★ 詳細CSVを保存する前に、サマリーを作成・保存 ---
    create_and_save_summaries(results, output_dir)

    # --- 詳細結果をカテゴリ別ディレクトリ & 特徴量タイプ別CSVに保存 ---
    print(f"\nSaving detailed results into category-specific directories...")
    for category, feature_data in results.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for feature_type, rows in feature_data.items():
            if not rows: continue
            
            df = pd.DataFrame(rows)
            csv_path = os.path.join(category_dir, f"{feature_type}_analysis.csv")
            df.to_csv(csv_path, index=False)
            print(f" -> Saved: {csv_path}")

    print(f"\n✅ Analysis complete. All files have been saved.")

# --- 5. メイン実行部 ---
def main():
    """スクリプトのメイン実行関数"""
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    print("\n--- Preparing model from Hugging Face ---")
    print(f"✅ Loading original pre-trained model: {MODEL_ID}")
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model and processor loaded successfully.")
    
    img_folder = os.path.join(DATA_DIR, 'val2017')
    ann_file_instances = os.path.join(DATA_DIR, 'annotations', 'instances_val2017.json')
    if not all(os.path.exists(f) for f in [img_folder, ann_file_instances]):
        print(f"Error: COCO dataset not found in {DATA_DIR}. Please check the path."); return
        
    dataset = CocoObjectDataset(
        root=img_folder, 
        annFile_instances=ann_file_instances, 
        categories=CATEGORIES_TO_ANALYZE, 
        num_samples_per_cat=NUM_SAMPLES_PER_CATEGORY,
        image_mode=INPUT_IMAGE_MODE,
        masked_img_dir=MASKED_IMAGE_DIR
    )
    
    run_combined_analysis(model, processor, dataset, OUTPUT_DIR)
    
    print("\n" + "-" * 50); print("✅ All processing is complete.")

if __name__ == "__main__":
    main()

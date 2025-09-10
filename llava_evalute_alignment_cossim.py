# -*- coding: utf-8 -*-
"""
VLM統合評価＆プロットスクリプト (v12: QAテストのエラー修正版)

本スクリプトの目的:
1. ★★★【新規】入力画像を「通常画像」と「マスク画像」から選択可能にする。★★★
2. ★★★【新規】モデル読み込みをHugging Faceからの直接ロードに一本化し、コードを簡潔にする。★★★
3. ★★★【新規】簡単なQAテストを実行するデバッグ機能を統合する。★★★
4. 評価結果から、目的別に分離された4種類のCSVファイルを出力し、グラフをプロットする。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import random
import datetime
import requests
from scipy.ndimage import binary_erosion

# --- 1. 設定 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DATASET_NAME = 'coco'
DATA_DIR = "./data"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# --- ▼▼▼ ここで実験設定を切り替え ▼▼▼ ---
# --- A. 実行モードの選択 ---
RUN_QA_TEST_ONLY = False # Trueにすると、下のB,Cの設定を無視して簡単なQAテストのみ実行します。

# --- B. 入力画像の選択 ---
INPUT_IMAGE_MODE = "normal"  # "normal" (通常画像) or "masked" (マスク処理画像) から選択

# --- C. 評価パラメータ ---
NUM_SAMPLES_PER_CATEGORY = 100
CATEGORIES_TO_ANALYZE = ["cat", "dog", "car", "bicycle", "person", "apple", "chair", "book"]
# --- ▲▲▲ 設定はここまで ▲▲▲ ---

# (以降のコードはv11と同じですが、QAテスト部分の1行のみ修正しています)
MASKED_IMAGE_DIR = os.path.join(DATA_DIR, "outline_masked_coco_val2017")
output_folder_name = f"{INPUT_IMAGE_MODE}_input"
OUTPUT_DIR = os.path.join("evaluation_results", DATASET_NAME, output_folder_name, TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)
if INPUT_IMAGE_MODE == "masked":
    os.makedirs(MASKED_IMAGE_DIR, exist_ok=True)

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

def get_embedding_for_word(tokenizer, word, embed_matrix):
    token_ids = tokenizer(word, add_special_tokens=False).input_ids
    if not token_ids: return None
    return embed_matrix[token_ids].mean(dim=0)

def get_all_features(image, model, processor):
    prompt = "USER: <image>\nASSISTANT:"
    inputs = processor.tokenizer(prompt, return_tensors="pt").to(DEVICE)
    image_tensor = processor.image_processor(image, return_tensors="pt")['pixel_values'].to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        vision_outputs = model.vision_tower(image_tensor, output_hidden_states=True)
        vision_last_hidden_state = vision_outputs.hidden_states[-1]
        projected_cls_feat = model.multi_modal_projector(vision_last_hidden_state[:, 0:1, :])
        projected_patch_feats = model.multi_modal_projector(vision_last_hidden_state[:, 1:, :])
        projected_all_feats = torch.cat([projected_cls_feat, projected_patch_feats], dim=1)
        mean_no_cls = projected_patch_feats.mean(dim=1)
        mean_with_cls = projected_all_feats.mean(dim=1)
        text_embeds = model.get_input_embeddings()(inputs.input_ids)
        image_token_index = torch.where(inputs.input_ids == model.config.image_token_index)[1]
        final_embeds = torch.cat([text_embeds[:, :image_token_index], projected_patch_feats, text_embeds[:, image_token_index + 1:]], dim=1)
        llm_outputs = model.language_model(inputs_embeds=final_embeds, output_hidden_states=True)
    llm_hidden_states = llm_outputs.hidden_states
    num_image_patches = projected_patch_feats.shape[1]
    image_start_index = image_token_index.item()
    unpooled_features_per_layer = [projected_patch_feats.cpu().float().squeeze(0)]
    for layer_hidden_state in llm_hidden_states[1:]:
        image_patch_features = layer_hidden_state[:, image_start_index : image_start_index + num_image_patches, :]
        unpooled_features_per_layer.append(image_patch_features.cpu().float().squeeze(0))
    return unpooled_features_per_layer, projected_cls_feat.cpu().float().squeeze(), mean_no_cls.cpu().float().squeeze(), mean_with_cls.cpu().float().squeeze()

def run_evaluation_and_get_data(model, processor, embed_matrix, dataset, llm_layers, categories, output_dir):
    print("-" * 50); print(f"🔍 Running evaluation and generating 4 CSV files...")
    # ... (この関数は変更なし)
    tokenizer = processor.tokenizer
    global_summary_results, pre_llm_patch_summary_results, llm_layer_summary_results, raw_similarity_results = [], [], [], []
    word_embeddings = {cat: get_embedding_for_word(tokenizer, cat, embed_matrix) for cat in categories}
    word_embeddings = {k: v for k, v in word_embeddings.items() if v is not None}
    word_embed_matrix = torch.stack(list(word_embeddings.values())); word_embed_matrix_norm = F.normalize(word_embed_matrix, p=2, dim=-1)
    word_list = list(word_embeddings.keys())
    for i in tqdm(range(len(dataset)), desc="Processing Images"):
        sample = dataset[i]; image, label, img_id = sample['image'], sample['label'], sample['img_id']
        if label not in word_list: continue
        label_idx = word_list.index(label)
        unpooled_features_per_layer, cls_feat, mean_no_cls_feat, mean_with_cls_feat = get_all_features(image, model, processor)
        if unpooled_features_per_layer is None: continue
        global_features = {'cls': cls_feat, 'mean_no_cls': mean_no_cls_feat, 'mean_with_cls': mean_with_cls_feat}
        for f_type, f_vec in global_features.items():
            all_sims = F.cosine_similarity(f_vec.unsqueeze(0), word_embed_matrix_norm, dim=1).numpy()
            pos_sim, neg_sims = all_sims[label_idx], np.delete(all_sims, label_idx)
            margin = pos_sim - np.max(neg_sims); top1_acc = 1 if np.argmax(all_sims) == label_idx else 0
            global_summary_results.append({"image_id": img_id, "label": label, "feature_type": f_type, "sim_pos": pos_sim, "margin": margin, "top1_acc": top1_acc})
            for j, word in enumerate(word_list): raw_similarity_results.append({'image_id': img_id, 'true_label': label, 'layer': 'pre_llm', 'feature_type': f_type, 'candidate_label': word, 'similarity_score': all_sims[j]})
        pre_llm_patch_features_norm = F.normalize(unpooled_features_per_layer[0], p=2, dim=-1)
        pre_llm_all_patch_sims = torch.matmul(pre_llm_patch_features_norm, word_embed_matrix_norm.T).numpy()
        pre_llm_max_sim_for_label = np.max(pre_llm_all_patch_sims[:, label_idx]); pre_llm_best_word_indices = np.argmax(pre_llm_all_patch_sims, axis=1)
        pre_llm_patch_level_accuracy = np.mean(pre_llm_best_word_indices == label_idx)
        pre_llm_patch_summary_results.append({"image_id": img_id, "label": label, "sim_pos_patch_max": pre_llm_max_sim_for_label, "patch_level_accuracy": pre_llm_patch_level_accuracy})
        max_sim_per_word_pre_llm = np.max(pre_llm_all_patch_sims, axis=0)
        for j, word in enumerate(word_list): raw_similarity_results.append({'image_id': img_id, 'true_label': label, 'layer': 0, 'feature_type': 'max_patch', 'candidate_label': word, 'similarity_score': max_sim_per_word_pre_llm[j]})
        for i, layer_idx in enumerate(llm_layers, 1):
            patch_features_norm = F.normalize(unpooled_features_per_layer[i], p=2, dim=-1)
            all_patch_sims = torch.matmul(patch_features_norm, word_embed_matrix_norm.T).numpy()
            max_sim_for_label = np.max(all_patch_sims[:, label_idx]); best_word_indices = np.argmax(all_patch_sims, axis=1)
            patch_level_accuracy = np.mean(best_word_indices == label_idx)
            llm_layer_summary_results.append({"image_id": img_id, "label": label, "layer": layer_idx, "sim_pos_patch_max": max_sim_for_label, "patch_level_accuracy": patch_level_accuracy})
            max_sim_per_word = np.max(all_patch_sims, axis=0)
            for j, word in enumerate(word_list): raw_similarity_results.append({'image_id': img_id, 'true_label': label, 'layer': layer_idx, 'feature_type': 'max_patch', 'candidate_label': word, 'similarity_score': max_sim_per_word[j]})
    df_global = pd.DataFrame(global_summary_results); df_pre_llm_patch = pd.DataFrame(pre_llm_patch_summary_results)
    df_llm_layer = pd.DataFrame(llm_layer_summary_results); df_raw = pd.DataFrame(raw_similarity_results)
    df_global.to_csv(os.path.join(output_dir, "pre_llm_global_summary.csv"), index=False)
    df_pre_llm_patch.to_csv(os.path.join(output_dir, "pre_llm_patch_summary.csv"), index=False)
    df_llm_layer.to_csv(os.path.join(output_dir, "llm_layer_summary.csv"), index=False)
    df_raw.to_csv(os.path.join(output_dir, "raw_similarity_scores.csv"), index=False)
    print(f"\n✅ Successfully saved 4 CSV files to {output_dir}")
    return df_global, df_pre_llm_patch, df_llm_layer, df_raw

def plot_all_summaries(df_global, df_pre_llm_patch, df_llm_layer, df_raw, output_dir, llm_layers):
    print("\n" + "-" * 50); print(f"📊 Plotting evaluation results...")
    # ... (この関数は変更なし)
    plot_dir = os.path.join(output_dir, "summary_plots"); os.makedirs(plot_dir, exist_ok=True)
    pre_llm_plot_dir = os.path.join(plot_dir, "pre_llm_analysis"); os.makedirs(pre_llm_plot_dir, exist_ok=True)
    llm_plot_dir = os.path.join(plot_dir, "llm_layer_analysis"); os.makedirs(llm_plot_dir, exist_ok=True)
    plot_global_performance(pre_llm_plot_dir, df_global)
    plot_category_difficulty(pre_llm_plot_dir, df_global, metric='margin')
    for f_type in ['cls', 'mean_no_cls', 'mean_with_cls']: plot_confusion_heatmap(pre_llm_plot_dir, df_raw, feature_type=f_type, layer='pre_llm')
    plot_confusion_heatmap(pre_llm_plot_dir, df_raw, feature_type='max_patch', layer=0)
    plot_patch_performance_over_layers(plot_dir, df_pre_llm_patch, df_llm_layer)
    for layer_idx in tqdm(llm_layers, desc="Plotting Layer-wise Heatmaps"): plot_confusion_heatmap(llm_plot_dir, df_raw, feature_type='max_patch', layer=layer_idx)
    print("✅ All plots have been generated successfully.")

def plot_global_performance(plot_dir, df_global):
    if df_global.empty: return
    # ... (この関数は変更なし)
    plt.figure(figsize=(18, 6)); 
    plt.subplot(1, 3, 1); sns.barplot(data=df_global, x='feature_type', y='sim_pos'); plt.title('Similarity to Ground-Truth'); plt.grid(True)
    plt.subplot(1, 3, 2); sns.barplot(data=df_global, x='feature_type', y='margin'); plt.title('Similarity Margin'); plt.grid(True)
    plt.subplot(1, 3, 3); sns.barplot(data=df_global, x='feature_type', y='top1_acc'); plt.title('Top-1 Accuracy'); plt.grid(True)
    plt.suptitle('Global Performance Summary (Pre-LLM)', fontsize=18); plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(plot_dir, "global_performance_summary.png"); plt.savefig(save_path, dpi=300); plt.close()
    print(f"-> Saved global performance plot: {save_path}")

def plot_patch_performance_over_layers(plot_dir, df_pre_llm_patch, df_llm_layer):
    if df_llm_layer.empty: return
    # ... (この関数は変更なし)
    df_pre_llm_patch_mean = df_pre_llm_patch.mean(numeric_only=True); df_pre_llm_patch_mean['layer'] = 0
    df_layer = pd.concat([pd.DataFrame([df_pre_llm_patch_mean]), df_llm_layer.groupby('layer').mean(numeric_only=True).reset_index()], ignore_index=True)
    fig, ax1 = plt.subplots(figsize=(14, 7)); ax2 = ax1.twinx()
    sns.lineplot(data=df_layer, x='layer', y='patch_level_accuracy', marker='o', color='g', ax=ax1, label='Patch-Level Accuracy')
    sns.lineplot(data=df_layer, x='layer', y='sim_pos_patch_max', marker='x', color='b', ax=ax2, label='Similarity to GT (Best Patch)')
    ax1.set_xlabel('LLM Layer Index (0=Pre-LLM)'); ax1.set_ylabel('Patch-Level Accuracy', color='g'); ax2.set_ylabel('Max Similarity', color='b'); ax1.tick_params(axis='y', labelcolor='g'); ax2.tick_params(axis='y', labelcolor='b')
    plt.title('Local Alignment Performance Across LLM Layers', fontsize=18); ax1.legend(loc='upper left'); ax2.legend(loc='upper right'); plt.xticks(df_layer['layer']); ax1.grid(True)
    save_path = os.path.join(plot_dir, "patch_performance_over_layers.png"); plt.savefig(save_path, dpi=300); plt.close()
    print(f"-> Saved patch performance plot: {save_path}")
    
def plot_category_difficulty(plot_dir, df_global, metric='margin'):
    if df_global.empty: return
    # ... (この関数は変更なし)
    plt.figure(figsize=(15, 8)); sns.barplot(data=df_global, x='label', y=metric, hue='feature_type')
    plt.title(f'Category Performance Report Card: {metric} (Pre-LLM)', fontsize=18); plt.xlabel('Category'); plt.ylabel(f'Average {metric}'); plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    save_path = os.path.join(plot_dir, f"category_difficulty_{metric}.png"); plt.savefig(save_path, dpi=300); plt.close()
    print(f"-> Saved category difficulty plot: {save_path}")

def plot_confusion_heatmap(plot_dir, df_raw, feature_type='max_patch', layer=0):
    if df_raw.empty: return
    # ... (この関数は変更なし)
    data_for_heatmap = df_raw[(df_raw['feature_type'] == feature_type) & (df_raw['layer'] == layer)]
    if data_for_heatmap.empty: print(f"-> No data for confusion matrix ({feature_type} @ Layer {layer}). Skipping."); return
    idx = data_for_heatmap.groupby(['image_id'])['similarity_score'].transform(max) == data_for_heatmap['similarity_score']
    predictions = data_for_heatmap[idx].drop_duplicates(subset=['image_id'])
    all_categories = sorted(df_raw['true_label'].unique())
    confusion_matrix = pd.crosstab(predictions['true_label'], predictions['candidate_label']).reindex(index=all_categories, columns=all_categories, fill_value=0)
    confusion_matrix_normalized = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0).fillna(0)
    plt.figure(figsize=(12, 10)); sns.heatmap(confusion_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(f'Confusion Matrix Heatmap: {feature_type} (at Layer {layer})', fontsize=18); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.tight_layout()
    save_path = os.path.join(plot_dir, f"confusion_heatmap_{feature_type}_layer_{layer}.png"); plt.savefig(save_path, dpi=300); plt.close()

def run_simple_qa_test():
    """簡単なQAテストを実行して、モデルの基本的な応答能力を確認する"""
    print("\n" + "-" * 20 + " DEBUGGING: SIMPLE QA TEST " + "-" * 20)
    try:
        print(f"QAテストのため、モデル '{MODEL_ID}' をロードします...")
        model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("モデルのロードが完了しました。")

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        prompt = "USER: <image>\nWhat is in the image?"

        # --- ▼▼▼ ここを修正 ▼▼▼ ---
        # 引数に名前(text=, images=)を明示的に指定
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, dtype=torch.float16)
        # --- ▲▲▲ 修正ここまで ▲▲▲ ---
        
        generate_ids = model.generate(**inputs, max_new_tokens=20)
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(f"\n[Image URL]: {url}")
        print(f"[Prompt]: {prompt}")
        print("-" * 50)
        print(f"[Generated Response]:\n{response}")
        print("-" * 50)

        response_lower = response.lower()
        if "cat" in response_lower or "feline" in response_lower:
            print("\n[✅] QA Test PASSED: The model correctly identified the object.")
        else:
            print("\n[❌] QA Test FAILED: The model could not identify the object.")
            print("      -> This strongly suggests a critical issue like a weight loading failure.")

    except Exception as e:
        print(f"\n[❌] An error occurred during the QA test: {e}")
    print("-" * 64 + "\n")

def main():
    """スクリプトのメイン実行関数"""
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    if RUN_QA_TEST_ONLY:
        run_simple_qa_test()
        return

    print("\n--- Preparing model from Hugging Face ---")
    print(f"✅ Loading original pre-trained model: {MODEL_ID}")
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model and processor loaded successfully.")
    
    llm_layers = list(range(1, len(model.language_model.layers) + 1))
    
    img_folder = os.path.join(DATA_DIR, 'val2017')
    ann_file_instances = os.path.join(DATA_DIR, 'annotations', 'instances_val2017.json')
    if not all(os.path.exists(f) for f in [img_folder, ann_file_instances]):
        print("Error: COCO dataset not found in ./data directory."); return
        
    dataset = CocoObjectDataset(
        root=img_folder, 
        annFile_instances=ann_file_instances, 
        categories=CATEGORIES_TO_ANALYZE, 
        num_samples_per_cat=NUM_SAMPLES_PER_CATEGORY,
        image_mode=INPUT_IMAGE_MODE,
        masked_img_dir=MASKED_IMAGE_DIR
    )
    
    embed_matrix = model.get_input_embeddings().weight.detach().cpu().float()
    df_global, df_pre_llm_patch, df_llm_layer, df_raw = run_evaluation_and_get_data(model, processor, embed_matrix, dataset, llm_layers, CATEGORIES_TO_ANALYZE, OUTPUT_DIR)
    
    if all(df is not None for df in [df_global, df_pre_llm_patch, df_llm_layer, df_raw]):
        plot_all_summaries(df_global, df_pre_llm_patch, df_llm_layer, df_raw, OUTPUT_DIR, llm_layers)
    
    print("\n" + "-" * 50); print("✅ All evaluations and plotting are complete.")

if __name__ == "__main__":
    main()
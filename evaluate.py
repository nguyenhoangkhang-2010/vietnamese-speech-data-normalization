import json
import os
from src.evaluation.wer import calculate_wer

def evaluate_results(manifest_path, ground_truth_dir):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    total_wer = 0
    for item in data:
        audio_name = os.path.basename(item["audio_filepath"]).replace(".mp4", "")
        gt_path = os.path.join(ground_truth_dir, f"{audio_name}.txt")
        
        if not os.path.exists(gt_path):
            print(f"LỖI: Không tìm thấy file gốc tại {gt_path}")
            continue
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_text = f.read().strip().lower()
            
        ai_text = item["text"].strip().lower()
        
        print(f"\n--- ĐANG ĐÁNH GIÁ: {audio_name} ---")
        print(f"Độ dài Gốc (từ): {len(gt_text.split())}")
        print(f"Độ dài AI (từ): {len(ai_text.split())}")
        
        wer = calculate_wer(gt_text, ai_text)
        print(f"WER: {wer:.2%}")
        
        print(f"Gốc (trích): {gt_text[:50]}...")
        print(f"AI (trích): {ai_text[:50]}...")
        
        total_wer += wer
        
    print(f"\n--- ĐÁNH GIÁ TỔNG KẾT ---")
    print(f"WER trung bình: {total_wer / len(data):.2%}")

if __name__ == "__main__":
    evaluate_results("data/processed/manifest.jsonl", "data/raw/transcripts")
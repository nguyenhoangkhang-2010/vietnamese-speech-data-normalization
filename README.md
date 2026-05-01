# Hệ thống Chuẩn hoá & Căn chỉnh Dữ liệu Giọng nói Tiếng Việt (NVIDIA NeMo Style)

## Tổng quan

Dự án này xây dựng một pipeline xử lý dữ liệu giọng nói tiếng Việt theo phong cách **NVIDIA NeMo**, giúp tạo dataset chất lượng cao cho các bài toán:

- Nhận dạng giọng nói (ASR)
- Tổng hợp giọng nói (TTS)
- Căn chỉnh audio - văn bản (Forced Alignment)
- Chuẩn hoá dữ liệu speech quy mô lớn

---

## Chức năng chính

### Text Normalization
- Chuẩn hoá văn bản tiếng Việt
- Xử lý số, ngày tháng, viết tắt
- Làm sạch dữ liệu text cho ASR

### Audio Processing
- Resample về 16kHz
- Chuẩn hoá biên độ
- Loại bỏ khoảng lặng (silence trimming)

### Alignment (Căn chỉnh)
- Căn audio với transcript
- Thiết kế theo hướng NeMo CTC / ASR alignment

### Dataset Builder
- Chuyển dữ liệu thô thành `manifest.jsonl`
- Tương thích NVIDIA NeMo training format

### Evaluation
- Tính Word Error Rate (WER)
- Thống kê chất lượng dataset

---

## Cấu trúc project
```text
├── configs
│   └── config.yaml
├── data
│   ├── processed
│   └── raw
│       ├── audio
│       └── transcripts
├── demo
│   └── run_demo.py
├── notebooks
│   └── data_exploration.ipynb
├── pipeline
│   └── pipeline.py
├── src
│   ├── alignment
│   │   └── aligner.py
│   ├── audio
│   │   ├── loader.py
│   │   ├── processor.py
│   │   └── validator.py
│   ├── dataset
│   │   └── builder.py
│   ├── evaluation
│   │   ├── metrics.py
│   │   ├── report.py
│   │   └── wer.py
│   ├── text
│   │   ├── rules
│   │   │   ├── abbreviation.py
│   │   │   ├── currency.py
│   │   │   ├── date.py
│   │   │   └── number.py
│   │   ├── normalizer.py
│   │   └── tokenizer.py
│   └── utils
│       ├── file_utils.py
│       └── logger.py
├── test
│   ├── test_alignment.py
│   ├── test_normalizer.py
│   └── test_pipeline.py
├── .gitignore
├── README.md
├── main.py
└── requirements.txt
```

## Cài đặt

### 1. Clone project

    git clone <your-repo-url>
    cd vietnamese-speech-nemo

### 2. Tạo môi trường ảo

    python -m venv .venv

Windows:
    
    .venv\Scripts\activate

Linux/Mac:
    
    source .venv/bin/activate

### 3. Cài dependencies

    pip install -r requirements.txt

---

## Chạy project

### Chạy toàn bộ pipeline

    python main.py

### Chạy demo

    python demo/run_demo.py

---

## Luồng xử lý dữ liệu

Raw Audio + Transcript  
→ Text Normalization (normalizer.py)  
→ Audio Processing (processor.py)  
→ Alignment (aligner.py - NeMo style)  
→ Dataset Builder (manifest.jsonl)  
→ Evaluation (WER + metrics)

---

## Các module chính

### src/text
- number.py
- date.py
- abbreviation.py
- currency.py

### src/audio
- loader
- processor
- validator

### src/alignment
- Forced alignment theo CTC (NeMo style)

### src/dataset
- Tạo manifest.jsonl chuẩn NeMo

### src/evaluation
- WER calculation
- Quality report

---

## notebooks

Dùng để:
- EDA dữ liệu
- Test normalization
- Debug alignment
- Thử nghiệm ASR model

---

## demo

- run_demo.py: chạy full pipeline
- demo_cli.py: test nhanh input

---

## Test

    pytest test/

---

## Requirements

- Python >= 3.13
- torch
- torchaudio
- nemo_toolkit[asr]
- librosa
- numpy
- jiwer
- pyyaml

---

## Hướng phát triển

- NeMo Conformer / QuartzNet integration
- Forced alignment có timestamp thật
- GPU batch processing
- Data augmentation
- Streamlit UI
- Docker deployment

---

## Tác giả

Nguyễn Hoàng Khang
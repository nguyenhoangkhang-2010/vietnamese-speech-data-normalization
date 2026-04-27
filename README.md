# Vietnamese Speech Data Normalization & Alignment System

## Overview
Dự án này xây dựng một hệ thống xử lý dữ liệu giọng nói tiếng Việt, bao gồm:
- Chuẩn hóa văn bản (text normalization)
- Tiền xử lý audio
- Căn chỉnh (alignment) giữa audio và transcript
- Đánh giá chất lượng dữ liệu

Ứng dụng:
- Speech Recognition (ASR)
- Text-to-Speech (TTS)
- Forced Alignment

---

## Project Structure

```text
data
│   ├── processed
│   └── raw
│       ├── audio
│       └── transcripts
│
├── demo
├── notebooks
├── pipeline
│
├── src
│   ├── alignment
│   │   └── aligner.py
│   ├── audio
│   ├── dataset
│   ├── evaluation
│   ├── text
│   └── utils
│
├── test
├── .gitignore
├── README.md
├── main.py
└── requirements.txt
```

---

## Installation

### 1. Clone repository
```bash
git clone <your-repo-url>
cd <your-project>
```

### 2. Tạo virtual environment
```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows
```

### 3. Cài dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Chạy chương trình chính
```bash
python main.py
```

### Chạy pipeline (nếu có)
```bash
python pipeline/run_pipeline.py
```

---

## Main Components

### Text Processing (`src/text`)
- Normalize text (lowercase, remove noise)
- Xử lý số, ngày tháng, ký tự đặc biệt

### Audio Processing (`src/audio`)
- Resampling
- Trim silence
- Noise handling

### Alignment (`src/alignment`)
- Căn chỉnh audio với transcript
- Có thể dùng CTC / MFA / custom

### Dataset (`src/dataset`)
- Load dữ liệu từ `data/raw`
- Chuẩn hóa format

### Evaluation (`src/evaluation`)
- Metrics đánh giá alignment

---

## Data Flow

```text
Raw Data → Text Processing → Audio Processing
         → Alignment → Processed Data → Evaluation
```

---

## Testing
```bash
pytest test/
```

---

## Requirements
- Python >= 3.8
- Xem `requirements.txt`

---

## Future Improvements
- [ ] Multi-speaker alignment
- [ ] Pretrained ASR integration
- [ ] Visualization tool
- [ ] Optimize performance

---

## Author
- Nguyễn Hoàng Khang
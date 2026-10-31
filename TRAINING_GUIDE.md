# Hướng Dẫn Chi Tiết Huấn Luyện 5 Biến Thể Aper (AdaPt and mERge) Trong Class-Incremental Learning

Tài liệu này cung cấp hướng dẫn toàn diện từ mã nguồn chính thức của bài báo:  
**"Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need"** (*IJCV 2024*).

---

## Mục Lục
1. [Tổng Quan Kiến Trúc & Cơ Chế Hoạt Động](#1-tổng-quan-kiến-trúc--cơ-chế-hoạt-động)
2. [Chi Tiết 5 Biến Thể Của Aper](#2-chi-tiết-5-biến-thể-của-aper)
3. [Cài Đặt Môi Trường Conda & Yêu Cầu Kỹ Thuật](#3-cài-đặt-môi-trường-conda--yêu-cầu-kỹ-thuật)
4. [Cấu Trúc & Tích Hợp 4 Tập Dữ Liệu Trong Thư Mục data/](#4-cấu-trúc--tích-hợp-4-tập-dữ-liệu-trong-thư-mục-data)
5. [Quy Tắc Cấu Hình Số Lượng Tasks (num_tasks)](#5-quy-tắc-cấu-hình-số-lượng-tasks-num_tasks)
6. [Mẫu Cấu Hình JSON Từng Biến Thể (exps/)](#6-mẫu-cấu-hình-json-từng-biến-thể-exps)
7. [Bộ Cấu Hình Tái Lập Chuẩn Hóa Tiện Lợi (scenarios/)](#7-bộ-cấu-hình-tái-lập-chuẩn-hóa-tiện-lợi-scenarios)
8. [Hướng Dẫn Thực Thi & Phân Tích Kết Quả Log](#8-hướng-dẫn-thực-thi--phân-tích-kết-quả-log)
9. [Script Tự Động Hóa Chạy Hàng Loạt (Bash Script)](#9-script-tự-động-hóa-chạy-hàng-loạt-bash-script)

---

## 1. Tổng Quan Kiến Trúc & Cơ Chế Hoạt Động

Class-Incremental Learning (CIL) truyền thống thường huấn luyện mạng từ đầu (from scratch), dễ dẫn đến hiện tượng quên thảm họa (Catastrophic Forgetting). Khi sử dụng các mô hình thị giác tiền huấn luyện lớn (Pre-Trained Models - PTM như ViT-B/16-IN21K), mạng đã sở hữu khả năng biểu diễn đặc trưng tổng quát cao (*Generalizability*).

Phương pháp **Aper (AdaPt and mERge)** được thiết kế theo quy trình 3 giai đoạn:

```
[Task 0 Dữ liệu] ──► Huấn luyện PEFT / Fine-tune ──► [Adapted ViT Backbone]
                                                             │
[Mô hình gốc PTM] ───────────────────────────────────────────┴──► [Dual-Branch Network]
                                                                        │ (Trích xuất đặc trưng nối: 768 + 768 = 1536 dim)
                                                                        ▼
[Tasks tiếp theo (t > 0)] ──► Tính Prototype Centroid không cần đạo hàm ──► [Cosine Classifier update]
```

1. **Stage 1 - Adaptation (Chỉ diễn ra ở Task 0):** Huấn luyện mô hình trên các lớp của task đầu tiên để hấp thu đặc trưng phân phối của downstream task (*Adaptivity*).
2. **Stage 2 - Merge (Hợp nhất Dual-Branch):** Đóng băng mô hình vừa học và ghép song song với một nhánh ViT tiền huấn luyện gốc (PTM gốc giữ *Generalizability*). Khi trích xuất đặc trưng, đầu ra của 2 nhánh được nối lại:
   $$\mathbf{f}(x) = [\mathbf{f}_{\text{PTM}}(x) \,\|\, \mathbf{f}_{\text{Adapted}}(x)] \in \mathbb{R}^{1536}$$
3. **Stage 3 - Training-Free Incremental Learning (Tasks $t \ge 1$):** Không thực hiện gradient descent, không cập nhật trọng số backbone. Với mỗi lớp mới, vector đại diện (Prototype) được tính bằng trung bình đặc trưng dual-branch của các mẫu trong lớp đó:
   $$\mathbf{p}_c = \frac{1}{|D_c|} \sum_{x \in D_c} \mathbf{f}(x)$$
   Vector này được chuẩn hóa và gán trực tiếp vào ma trận trọng số của bộ phân loại `CosineLinear`.

---

## 2. Chi Tiết 5 Biến Thể Của Aper

| Biến thể | File Model | Backbone `convnet_type` | Cơ chế điều chỉnh ở Task 0 |
| :--- | :--- | :--- | :--- |
| **Aper-Finetune** | `models/aper_finetune.py` | `pretrained_vit_b16_224_in21k` | Cập nhật toàn bộ trọng số của backbone ViT. |
| **Aper-VPT-Shallow** | `models/aper_vpt.py` | `pretrained_vit_b16_224_in21k_vpt` | Chỉ chèn thêm các prompt tokens có thể học vào **Transformer Layer đầu tiên (Layer 0)**; đóng băng toàn bộ backbone. |
| **Aper-VPT-Deep** | `models/aper_vpt.py` | `pretrained_vit_b16_224_in21k_vpt` | Chèn prompt tokens độc lập vào **tất cả 12 Transformer Layers**; đóng băng backbone. |
| **Aper-Adapter** | `models/aper_adapter.py` | `pretrained_vit_b16_224_in21k_adapter` | Chèn module AdaptFormer song song với khối MLP/FFN của mỗi block ViT; chỉ học trọng số của Adapter. |
| **Aper-SSF** | `models/aper_ssf.py` | `pretrained_vit_b16_224_in21k_ssf` | Chèn tham số Scale ($\gamma$) và Shift ($\beta$) tuyến tính sau mỗi phép biến đổi ma trận; chỉ học $\gamma$ và $\beta$. |

---

## 3. Cài Đặt Môi Trường Conda & Yêu Cầu Kỹ Thuật

> [!CAUTION]
> **Cảnh báo tính tương thích phiên bản thư viện:**
> Repo phụ thuộc trực tiếp vào các hàm nội bộ của `timm` cũ (`timm.models.helpers` bao gồm `to_2tuple`, `resolve_pretrained_cfg`, `build_model_with_cfg`). Nếu cài `timm >= 0.9` hoặc `timm 1.0`, chương trình sẽ báo lỗi `ModuleNotFoundError: No module named 'timm.models.layers.helpers'`. Cần tuân thủ đúng phiên bản bên dưới.

### Bước 3.1: Tạo Conda Environment
```bash
conda create -n aper python=3.9 -y
conda activate aper
```

### Bước 3.2: Cài đặt PyTorch và CUDA
Khuyến nghị sử dụng PyTorch 1.11.0:

- **Với CUDA 11.3 (qua Conda):**
  ```bash
  conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
  ```

- **Hoặc qua Wheel Pip (nhanh và tương thích tốt từ CUDA 11.3 đến 11.8):**
  ```bash
  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
  ```

### Bước 3.3: Cài đặt Dependencies
File `requirements.txt` đã được cấu hình trong thư mục gốc:
```bash
pip install timm==0.6.12 easydict scipy tqdm Pillow numpy
```

---

## 4. Cấu Trúc & Tích Hợp 4 Tập Dữ Liệu Trong Thư Mục data/

Thư mục `data/` trong workspace có cấu trúc thực tế như sau:

```
data/
├── CIFAR100/
│   ├── train/  (100 thư mục lớp, mỗi lớp 500 ảnh .png)
│   └── test/   (100 thư mục lớp, mỗi lớp 100 ảnh .png)
├── CUB200/
│   ├── train/  (200 thư mục lớp chim)
│   └── test/   (200 thư mục lớp chim)
├── imagenet-a/
│   ├── train/  (200 thư mục lớp ranh giới khó)
│   └── test/   (200 thư mục lớp ranh giới khó)
└── imagenet-r/
    ├── train/  (200 thư mục lớp art/cartoon)
    └── test/   (200 thư mục lớp art/cartoon)
```

### Cách nhận diện dữ liệu trong mã nguồn (`utils/data.py`):
Mã nguồn đã được thiết lập để tự động tương thích:
1. **CIFAR-100 (tên dataset trong JSON: `"cifar224"`):** Code ưu tiên nạp trực tiếp qua `datasets.ImageFolder` từ thư mục `./data/CIFAR100/train` và `test` mà không cần kết nối mạng để tải lại file nén.
2. **CUB-200 (tên dataset trong JSON: `"cub"`):** Code tự động kiểm tra thư mục `./data/CUB200/` hoặc `./data/cub/`.
3. **ImageNet-A (tên dataset trong JSON: `"imageneta"`):** Đọc trực tiếp từ `./data/imagenet-a/`.
4. **ImageNet-R (tên dataset trong JSON: `"imagenetr"`):** Đọc trực tiếp từ `./data/imagenet-r/`.

---

## 5. Quy Tắc Cấu Hình Số Lượng Tasks (num_tasks)

Số lượng tasks trong Aper được định nghĩa bởi 2 giá trị trong file cấu hình JSON:
- `init_cls`: Số lượng lớp học ở Task 0.
- `increment`: Số lượng lớp bổ sung ở mỗi task tiếp theo ($t = 1, 2, ...$).

Công thức tính tổng số tasks $T$:
$$T = 1 + \left\lceil \frac{C_{\text{total}} - init\_cls}{increment} \right\rceil$$

### 5.1. Bảng Tra Cứu Cho CIFAR-100 ($C_{\text{total}} = 100$ lớp)

| Số Task ($T$) | Kịch bản phân chia | `init_cls` | `increment` |
| :---: | :--- | :---: | :---: |
| **5 Tasks** | 5 tasks đều (20 lớp/task) | `20` | `20` |
| **10 Tasks** | 10 tasks đều (10 lớp/task) | `10` | `10` |
| **20 Tasks** | 20 tasks đều (5 lớp/task) | `5` | `5` |
| **6 Tasks** | Base-50 (50 lớp đầu, 5 tasks tiếp theo mỗi task 10 lớp) | `50` | `10` |
| **11 Tasks** | Base-50 (50 lớp đầu, 10 tasks tiếp theo mỗi task 5 lớp) | `50` | `5` |

### 5.2. Bảng Tra Cứu Cho CUB-200, ImageNet-A, ImageNet-R ($C_{\text{total}} = 200$ lớp)

| Số Task ($T$) | Kịch bản phân chia | `init_cls` | `increment` |
| :---: | :--- | :---: | :---: |
| **5 Tasks** | 5 tasks đều (40 lớp/task) | `40` | `40` |
| **10 Tasks** | 10 tasks đều (20 lớp/task) | `20` | `20` |
| **20 Tasks** | 20 tasks đều (10 lớp/task) | `10` | `10` |
| **40 Tasks** | 40 tasks đều (5 lớp/task) | `5` | `5` |
| **11 Tasks** | Base-100 (100 lớp đầu, 10 tasks tiếp theo mỗi task 10 lớp) | `100` | `10` |

---

## 6. Mẫu Cấu Hình JSON Từng Biến Thể (exps/)

> [!IMPORTANT]
> **Các cấu hình đã được CỐ ĐỊNH (FIXED) cho mọi experiment:**
> - `tuned_epoch`: **20** (Số epoch huấn luyện ở task ban đầu)
> - `optimizer`: **"sgd"**
> - `init_lr`: **0.05** (Tốc độ học khởi tạo)
> - `batch_size`: **32**
> 
> Bốn giá trị trên đã được cấu hình cố định trong mã nguồn [main.py](file:///home/hieunq10/CILBenchmarks/CIL-Aper/main.py) và đồng bộ trong toàn bộ các file cấu hình tại thư mục `exps/`. Các tham số còn lại (`dataset`, `init_cls`, `increment`, `device`, `seed`, `convnet_type`,...) có thể tùy biến tự do theo nhu cầu.

### 6.1. Biến Thể 1: Aper-Finetune
File ví dụ: `exps/aper_finetune/custom_run.json`
```json
{
    "prefix": "exp_finetune",
    "dataset": "imagenetr",
    "shuffle": true,
    "init_cls": 20,
    "increment": 20,
    "model_name": "aper_finetune",
    "convnet_type": "pretrained_vit_b16_224_in21k",
    "optimizer": "sgd",
    "init_lr": 0.05,
    "weight_decay": 0.0005,
    "min_lr": 0,
    "tuned_epoch": 20,
    "batch_size": 32,
    "device": ["0"],
    "seed": [1993]
}
```

### 6.2. Biến Thể 2: Aper-VPT-Shallow
File ví dụ: `exps/aper_vpt-shallow/custom_run.json`
```json
{
    "prefix": "exp_vpt-shallow",
    "dataset": "imagenetr",
    "shuffle": true,
    "init_cls": 20,
    "increment": 20,
    "model_name": "aper_vpt",
    "convnet_type": "pretrained_vit_b16_224_in21k_vpt",
    "vpt_type": "shallow",
    "prompt_token_num": 5,
    "ffn_num": 64,
    "optimizer": "sgd",
    "init_lr": 0.05,
    "weight_decay": 0.0005,
    "min_lr": 0,
    "tuned_epoch": 20,
    "batch_size": 32,
    "device": ["0"],
    "seed": [1993]
}
```

### 6.3. Biến Thể 3: Aper-VPT-Deep
File ví dụ: `exps/aper_vpt-deep/custom_run.json`
```json
{
    "prefix": "exp_vpt-deep",
    "dataset": "imagenetr",
    "shuffle": true,
    "init_cls": 20,
    "increment": 20,
    "model_name": "aper_vpt",
    "convnet_type": "pretrained_vit_b16_224_in21k_vpt",
    "vpt_type": "deep",
    "prompt_token_num": 5,
    "ffn_num": 64,
    "optimizer": "sgd",
    "init_lr": 0.05,
    "weight_decay": 0.0005,
    "min_lr": 0,
    "tuned_epoch": 20,
    "batch_size": 32,
    "device": ["0"],
    "seed": [1993]
}
```

### 6.4. Biến Thể 4: Aper-Adapter (AdaptFormer)
File ví dụ: `exps/aper_adapter/custom_run.json`
```json
{
    "prefix": "exp_adapter",
    "dataset": "imagenetr",
    "shuffle": true,
    "init_cls": 20,
    "increment": 20,
    "model_name": "aper_adapter",
    "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
    "prompt_token_num": 30,
    "ffn_num": 64,
    "optimizer": "sgd",
    "init_lr": 0.05,
    "weight_decay": 0.005,
    "min_lr": 0,
    "tuned_epoch": 20,
    "batch_size": 32,
    "device": ["0"],
    "seed": [1993]
}
```

### 6.5. Biến Thể 5: Aper-SSF
File ví dụ: `exps/aper_ssf/custom_run.json`
```json
{
    "prefix": "exp_ssf",
    "dataset": "imagenetr",
    "shuffle": true,
    "init_cls": 20,
    "increment": 20,
    "model_name": "aper_ssf",
    "convnet_type": "pretrained_vit_b16_224_in21k_ssf",
    "prompt_token_num": 30,
    "ffn_num": 64,
    "optimizer": "sgd",
    "init_lr": 0.05,
    "weight_decay": 0.0005,
    "min_lr": 0,
    "tuned_epoch": 20,
    "batch_size": 32,
    "device": ["0"],
    "seed": [1993]
}
```

---

## 7. Bộ Cấu Hình Tái Lập Chuẩn Hóa Tiện Lợi (scenarios/)

Để phục vụ mục đích **tái lập thực nghiệm (Reproduce)** một cách tiện lợi và tối giản nhất, thư mục `scenarios/` cung cấp sẵn các file cấu hình gộp. Mỗi file đại diện cho một kịch bản hoàn chỉnh, tích hợp sẵn **toàn bộ 5 biến thể của Aper** và thực thi trên **3 seeds ngẫu nhiên** (`0`, `1`, `2`):

### 7.1. Danh Mục 6 Kịch Bản Thực Nghiệm

| STT | Kịch bản thực nghiệm | Tập dữ liệu | Số Tasks & Số Lớp | File cấu hình trong `scenarios/` | File alias ngắn gọn |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | CIFAR-100 (10 tasks) | CIFAR-100 (`cifar224`) | 10 tasks, mỗi task 10 lớp | `scenarios/cifar100_10tasks_10cls.json` | `scenarios/cifar100_10_10.json` |
| **2** | CUB-200 (10 tasks) | CUB-200 (`cub`) | 10 tasks, mỗi task 20 lớp | `scenarios/cub200_10tasks_20cls.json` | `scenarios/cub200_10_20.json` |
| **3** | ImageNet-R (10 tasks) | ImageNet-R (`imagenetr`) | 10 tasks, mỗi task 20 lớp | `scenarios/imagenetr_10tasks_20cls.json` | `scenarios/imagenetr_10_20.json` |
| **4** | ImageNet-A (10 tasks) | ImageNet-A (`imageneta`) | 10 tasks, mỗi task 20 lớp | `scenarios/imageneta_10tasks_20cls.json` | `scenarios/imageneta_10_20.json` |
| **5** | ImageNet-R (20 tasks) | ImageNet-R (`imagenetr`) | 20 tasks, mỗi task 10 lớp | `scenarios/imagenetr_20tasks_10cls.json` | `scenarios/imagenetr_20_10.json` |
| **6** | ImageNet-R (40 tasks) | ImageNet-R (`imagenetr`) | 40 tasks, mỗi task 5 lớp | `scenarios/imagenetr_40tasks_5cls.json` | `scenarios/imagenetr_40_5.json` |

### 7.2. Cấu Trúc File Cấu Hình Kịch Bản (Scenario JSON)
Mỗi file JSON gom các thiết lập chung (tập dữ liệu, quy cách chia task, danh sách seeds, bộ nhớ replay) và khai báo danh sách 5 biến thể của Aper trong trường `"models"`:
1. `aper_finetune` (Full fine-tuning ViT)
2. `aper_vpt` shallow (Visual Prompt Tuning ở Layer 0, 5 tokens)
3. `aper_vpt` deep (Visual Prompt Tuning ở cả 12 Layers, 5 tokens)
4. `aper_adapter` (AdaptFormer ViT, `ffn_num=64`)
5. `aper_ssf` (Scale & Shift Features, `ffn_num=64`)

Ví dụ nội dung file `scenarios/cifar100_10_10.json`:
```json
{
    "scenario_name": "cifar100_10tasks_10cls",
    "dataset": "cifar224",
    "shuffle": true,
    "init_cls": 10,
    "increment": 10,
    "memory_size": 2000,
    "memory_per_class": 20,
    "fixed_memory": false,
    "device": ["0"],
    "seed": [0, 1, 2],
    "models": [
        {
            "model_name": "aper_finetune",
            "prefix": "exp_finetune",
            "convnet_type": "pretrained_vit_b16_224_in21k",
            "weight_decay": 0.0005,
            "min_lr": 0
        },
        {
            "model_name": "aper_vpt",
            "prefix": "exp_vpt-shallow",
            "convnet_type": "pretrained_vit_b16_224_in21k_vpt",
            "vpt_type": "shallow",
            "prompt_token_num": 5,
            "weight_decay": 0.0005,
            "min_lr": 0
        },
        {
            "model_name": "aper_vpt",
            "prefix": "exp_vpt-deep",
            "convnet_type": "pretrained_vit_b16_224_in21k_vpt",
            "vpt_type": "deep",
            "prompt_token_num": 5,
            "weight_decay": 0.0005,
            "min_lr": 0
        },
        {
            "model_name": "aper_adapter",
            "prefix": "exp_adapter",
            "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
            "prompt_token_num": 30,
            "ffn_num": 64,
            "weight_decay": 0.0005,
            "min_lr": 0
        },
        {
            "model_name": "aper_ssf",
            "prefix": "exp_ssf",
            "convnet_type": "pretrained_vit_b16_224_in21k_ssf",
            "prompt_token_num": 30,
            "ffn_num": 64,
            "weight_decay": 0.0005,
            "min_lr": 0
        }
    ]
}
```

### 7.3. Hướng Dẫn Chạy Một Lệnh Duy Nhất (Single Command)

Chỉ với 1 lệnh, hệ thống sẽ tự động chạy toàn bộ **5 biến thể Aper $\times$ 3 seeds = 15 thực nghiệm**:
```bash
# Tái lập kịch bản CIFAR-100 (10 tasks, 10 classes)
python main.py --config scenarios/cifar100_10_10.json

# Tái lập kịch bản CUB-200 (10 tasks, 20 classes)
python main.py --config scenarios/cub200_10_20.json

# Tái lập kịch bản ImageNet-R (10 tasks, 20 classes)
python main.py --config scenarios/imagenetr_10_20.json

# Tái lập kịch bản ImageNet-A (10 tasks, 20 classes)
python main.py --config scenarios/imageneta_10_20.json

# Tái lập kịch bản ImageNet-R (20 tasks, 10 classes)
python main.py --config scenarios/imagenetr_20_10.json

# Tái lập kịch bản ImageNet-R (40 tasks, 5 classes)
python main.py --config scenarios/imagenetr_40_5.json
```

### 7.4. Các Tham Số Dòng Lệnh Nâng Cao (Tùy Chọn)
- **Chỉ định GPU trực tiếp:** Thêm `--device <id>` (ví dụ `--device 0` hoặc `--device 1`) để ghi đè GPU mà không cần sửa file JSON:
  ```bash
  python main.py --config scenarios/imagenetr_10_20.json --device 0
  ```
- **Lọc chạy riêng 1 biến thể cụ thể:** Thêm `--model <tên_model_hoặc_prefix>`:
  ```bash
  # Chỉ chạy Aper-SSF trên cả 3 seeds
  python main.py --config scenarios/imagenetr_10_20.json --model aper_ssf

  # Chỉ chạy Aper-VPT-Deep trên cả 3 seeds
  python main.py --config scenarios/imagenetr_10_20.json --model exp_vpt-deep
  ```
- **Tùy chỉnh danh sách seeds:** Thêm `--seed <seed1,seed2,...>`:
  ```bash
  python main.py --config scenarios/cifar100_10_10.json --seed 0,1,2
  ```

---

## 8. Hướng Dẫn Thực Thi & Phân Tích Kết Quả Log

### 8.1. Lệnh Khởi Chạy Cho File Đơn Lẻ (exps/)
Chạy qua file `main.py` với đối số `--config`:
```bash
python main.py --config ./exps/aper_adapter/cifar224_5_5_vit-b_adapter.json
```

Để chỉ định GPU cụ thể khi máy có nhiều card:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config ./exps/aper_ssf/imagenetr_5_5_vit-b_ssf.json
```

### 8.2. Phân Tích Kết Quả Log
Log thực thi được tự động lưu theo cấu trúc thư mục:
`logs/{model_name}/{dataset}/{init_cls}/{increment}/{prefix}_{seed}_{convnet_type}.log`

Ví dụ một đoạn log chuẩn khi hoàn thành 1 task:
```text
2024-08-24 17:07:55 [aper_finetune.py] => Learning on 0-5
2024-08-24 17:18:53 [aper_finetune.py] => Task 0, Epoch 20/20 => Loss 0.122, Train_accy 96.56, Test_accy 98.20
...
2024-08-24 19:17:49 [trainer.py] => CNN top1 curve: [97.8, 94.9, 93.93, 92.2, 92.08, 91.77, 89.77, 89.08, ..., 81.27]
2024-08-24 19:17:49 [trainer.py] => Average Accuracy (CNN): 87.558
```
- `CNN top1 curve`: Danh sách độ chính xác kiểm tra sau mỗi task từ $0$ đến $T-1$.
- `Average Accuracy (CNN)`: Độ chính xác trung bình tích lũy $\bar{A} = \frac{1}{T} \sum_{i=1}^T A_i$ (tiêu chuẩn đánh giá chính trong các bài báo CIL).

---

## 9. Script Tự Động Hóa Chạy Hàng Loạt (Bash Script)

Tạo file `run_experiments.sh` tại thư mục gốc để tự động chạy cả 5 biến thể trên tập dữ liệu mong muốn:

```bash
#!/usr/bin/env bash
# ==============================================================================
# Script tự động chạy 5 biến thể của Aper
# ==============================================================================
set -e

# Kích hoạt môi trường conda (điều chỉnh đường dẫn miniconda nếu cần)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aper

export CUDA_VISIBLE_DEVICES=0

DATASET="imagenetr"
echo "Bắt đầu chuỗi thực nghiệm trên tập dữ liệu: $DATASET"

# 1. Aper-Finetune
echo "--------------------------------------------------------"
echo ">>> [1/5] Chạy Aper-Finetune..."
python main.py --config ./exps/aper_finetune/${DATASET}_5_5_vit-b_finetune.json

# 2. Aper-VPT-Shallow
echo "--------------------------------------------------------"
echo ">>> [2/5] Chạy Aper-VPT-Shallow..."
python main.py --config ./exps/aper_vpt-shallow/${DATASET}_5_5_vit-b_vpt-shallow.json

# 3. Aper-VPT-Deep
echo "--------------------------------------------------------"
echo ">>> [3/5] Chạy Aper-VPT-Deep..."
python main.py --config ./exps/aper_vpt-deep/${DATASET}_5_5_vit-b_vpt-deep.json

# 4. Aper-Adapter
echo "--------------------------------------------------------"
echo ">>> [4/5] Chạy Aper-Adapter..."
python main.py --config ./exps/aper_adapter/${DATASET}_5_5_vit-b_adapter.json

# 5. Aper-SSF
echo "--------------------------------------------------------"
echo ">>> [5/5] Chạy Aper-SSF..."
python main.py --config ./exps/aper_ssf/${DATASET}_5_5_vit-b_ssf.json

echo "========================================================"
echo "Hoàn thành toàn bộ 5 biến thể của Aper!"
```

Cấp quyền thực thi và chạy ngầm (background):
```bash
chmod +x run_experiments.sh
nohup ./run_experiments.sh > run_experiments.log 2>&1 &
```
Theo dõi tiến trình chạy trực tiếp:
```bash
tail -f run_experiments.log
```

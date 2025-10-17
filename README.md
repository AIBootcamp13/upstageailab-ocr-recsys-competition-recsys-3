# Title (Please modify the title)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- OS: Ubuntu 20.04.6 LTS  
- Python: 3.10 (venv 권장)  
- GPU: CUDA 환경 선택적(있으면 SASRec 학습/추론 가속)  
- Shell: bash / zsh

### Requirements
> 핵심 의존성은 **버전 고정** 권장
- Python pkgs: `numpy==1.26.4`, `pandas`, `pyarrow`, `fastparquet`, `torch`(CUDA 유무에 맞춰 설치), `recbole`, `implicit`, `scipy`, `tqdm`
- System pkgs: `build-essential`, `libopenblas-dev`, `gfortran`

## 1. Competiton Info

### Overview
- 온라인 스토어의 사용자 행동 로그(view / cart / purchase)를 이용해 **사용자별 Top-K 추천**을 생성하는 프로젝트입니다.  
베이스라인으로 **ALS (implicit MF)**, **SASRec (Sequential Transformer, Recbole)** 를 구축했고, 선택적으로 **RRF/Hybrid 앙상블**과 간단한 **재랭킹(co-visitation, 브랜드/카테고리 일치)** 를 적용합니다.

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview
- 파일: `train.parquet` (약 **8,350,311** 행)
- 주요 컬럼:
  - `user_id`, `item_id`, `user_session`
  - `event_time` (UTC)
  - `event_type` ∈ {`view`, `cart`, `purchase`}
  - `category_code`, `brand`, `price`
- 특성: 고희소(sparse)·롱테일 → **시퀀스/최근성** + **잠재요인** 신호 모두 중요

### EDA
- 이벤트 분포: `view` 다수, `cart`/`purchase` 비중은 낮지만 **신호 강도** 높음
- 시계열/세션: 최근 상호작용의 영향이 큼 → 시퀀스 모델(SASRec) 유리
- 인기 편향: 일부 상위 아이템 집중 → ALS로 전역 잠재 패턴 보완 필요
- 결측/이상치: 주요 필드 결측은 드묾(모델 전 처리 단계에서 최소 정제)

### Data Processing
- 공통
  - 시간 기준 정렬: (`user_session`, `event_time`)
  - 인덱싱: `user2idx.json`, `item2idx.json` 생성
- Recbole(SASRec)용 포맷
  - 스크립트: `code/recbole_dataset.py`
  - 산출: `../data/SASRec_dataset/SASRec_dataset.inter`
  - 필드: `user_idx:token`, `item_idx:token`, `event_time:float`
- ALS용 행렬
  - 스크립트: `code/train_als.py`
  - 사용자–아이템 CSR 생성
  - (선택) **이벤트 가중**(view<cart<purchase), **시간 감쇠** `exp(-λ·age_days)` 반영

## 4. Modeling

### Model descrition
- **AlternatingLeastSquares (implicit MF)**
  - 라이브러리: `implicit`
  - 베이스라인 하이퍼: `factors=32`, `alpha=10`, `regularization=0.001`
  - 장점: 전역 잠재요인/공출현 패턴에 강함, 학습·추론 속도 우수
  - 개선: 이벤트/최근성 가중으로 implicit 신호 품질 향상
- **SASRec (Sequential Transformer)**
  - 라이브러리: `recbole`
  - 예시 구성: `hidden_size=64`, `num_layers=2`, `max_seq_length≈50`, `BPRLoss`
  - 장점: 시간/순서(최근성·문맥) 신호 직접 학습
- **Ensemble (선택)**
  - **RRF**: `1/(α + rank)` (α=40~80 권장)
  - **Hybrid**: `λ·RRF + (1-λ)·(정규화 score)`; 모델 가중은 `w_sasrec > w_als` 소폭 우위
- **Re-ranking (선택)**
  - 사용자 최근 `cart/purchase`의 **브랜드/카테고리 일치** 보너스
  - **Co-visitation**(세션 공출현, 시간 감쇠) 점수 가산

### Modeling Process
```
# 1) 환경
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
pip install numpy==1.26.4 pyarrow fastparquet implicit recbole

# 2) Recbole 포맷 생성(SASRec)
python code/recbole_dataset.py --data_dir ../data --train_dataset train.parquet
# -> ../data/SASRec_dataset/SASRec_dataset.inter

# 3) SASRec 학습 / 추론
python code/train_sasrec.py --config_file code/yaml/sasrec.yaml
python code/inference_sasrec.py --data_dir ../data --output_dir ../output \
  --model_file ./saved/<SASRec-*.pth>
# -> ../output/submission_sasrec.csv

# 4) ALS 학습 / 추론
python code/train_als.py --dir_path ../data --data_dir train.parquet --output_dir ../output
# -> ../output/submission_als.csv

# 5) (선택) 앙상블
python code/ensemble_rrf_quick.py \
  --als ../output/submission_als.csv \
  --sasrec ../output/submission_sasrec.csv \
  --save_path ../output/submission_ensemble.csv \
  --alpha 60 --w_als 1.0 --w_sasrec 1.2 --topk 10
# -> ../output/submission_ensemble.csv
```

## 5. Result

### Leader Board

- NDCG@10 : 0.1154 (Rank 1)

### Presentation

- [발표자료_](https://docs.google.com/presentation/d/1wsUXjvP5cjk1BDtX93XkvK36gyHheJy5/edit?slide=id.p1#slide=id.p1)


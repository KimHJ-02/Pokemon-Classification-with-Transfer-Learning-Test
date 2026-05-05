# Pokemon Classification with Transfer Learning

주어진 포켓몬 이미지에서 포켓몬 이름(클래스)을 분류하는 프로젝트입니다.  
Kaggle `7,000 Labeled Pokemon` 데이터를 사용해 4개 이상의 실험 설정으로 성능을 비교하고, Streamlit GUI로 업로드 이미지 예측 데모를 제공합니다.

## 1) Objective

- 포켓몬 이미지 분류기 구현
- 4개 이상 실험 설정(백본/사전학습 사용 여부 등) 비교
- Precision / Recall / F1 등 지표 문서화
- GUI에서 테스트 이미지 업로드 및 모델 예측 확인

## 2) Dataset

- Source: Kaggle - `lantian773030/pokemonclassification`
- Number of classes: 150
- 스크립트로 자동 준비되는 구조:
  - `data/train/<class_name>/*.jpg`
  - `data/val/<class_name>/*.jpg`

## 3) Project Files

- `download_data.py`: 데이터 다운로드 및 `train/val` 구조 생성
- `dataset.py`: 데이터셋 로딩(`PokemonDataset`), 클래스 이름 로딩
- `models.py`: 모델 생성 헬퍼(`resnet50`, `mobilenet_v2`, `efficientnet_b0`, `simplecnn`)
- `train.py`: 단일 모델 학습
- `evaluate.py`: 단일 모델 평가 및 `classification_report.csv` 저장
- `run_experiments.py`: 4개 모델 학습/평가 자동 실행
- `streamlit_app.py`: 이미지 업로드 기반 GUI 데모

## 4) Environment Setup

```bash
pip install -r requirements.txt
```

권장: Python 3.9+

## 5) How To Run

### (1) Download and prepare data

```bash
python download_data.py
```

### (2) Run all experiments (4 models)

```bash
python run_experiments.py
```

실행 시 각 모델의 로그가 `logs/*.log`에 저장되고, 터미널에도 실시간 출력됩니다.

### (3) Train single model (optional)

```bash
python train.py --model resnet50 --data_dir data --epochs 5 --batch_size 32 --pretrained True
```

### (4) Evaluate single checkpoint (optional)

```bash
python evaluate.py --checkpoint checkpoints/resnet50_best.pth --model resnet50 --data_dir data
```

### (5) Run GUI demo

```bash
streamlit run streamlit_app.py
```

브라우저에서 업로드 이미지를 넣으면 4개 모델의 top-1 예측 결과를 각각 확인할 수 있습니다.

## 6) Experiment Settings

`run_experiments.py` 기본 실험 설정:

- ResNet50 (`pretrained=True`)
- MobileNetV2 (`pretrained=True`)
- EfficientNet-B0 (`pretrained=True`)
- SimpleCNN (`pretrained=False`)

필요 시 epoch, batch size, lr 등을 수정해 추가 실험을 진행할 수 있습니다.

## 7) Outputs

- 체크포인트: `checkpoints/<model>_best.pth`
- 학습 로그: `logs/<model>.log`
- 평가 로그: `logs/eval_<model>.log`
- 학습 곡선 원본: `history_<model>.csv`
- 분류 리포트: `classification_report_<model>.csv`

## 8) Result Summary

- Best model: `efficientnet_b0`
- Best validation accuracy: `0.9168` (epoch 3)
- 각 모델 성능 요약:
  - `efficientnet_b0`: accuracy `0.9168`, macro precision `0.9277`, macro recall `0.9155`, macro f1 `0.9129`
  - `mobilenet_v2`: accuracy `0.9140`, macro precision `0.9247`, macro recall `0.9132`, macro f1 `0.9098`
  - `resnet50`: accuracy `0.8909`, macro precision `0.9051`, macro recall `0.8871`, macro f1 `0.8841`
  - `simplecnn`: accuracy `0.0490`, macro precision `0.0243`, macro recall `0.0415`, macro f1 `0.0204`
- 모델별 비교 해석:
  - Transfer learning을 사용한 모델(`efficientnet_b0`, `mobilenet_v2`, `resnet50`)이 custom `simplecnn` 대비 매우 높은 성능을 보였습니다.
  - 본 실험(5 epoch)에서는 `efficientnet_b0`가 가장 높은 검증 정확도와 macro F1을 달성했습니다.
  - `simplecnn`은 클래스 수(150) 대비 표현력이 부족하고 수렴이 느려 성능이 크게 낮았습니다.
- 오분류 경향(보고서 해석):
  - 상위 모델들도 일부 클래스에서 recall 저하가 나타나며, 시각적으로 유사한 포켓몬 간 혼동 가능성이 남아 있습니다.
  - 더 긴 학습, data augmentation 강화, class-balanced sampling 적용 시 추가 개선 여지가 있습니다.

## 9) Learning Curve

- `history_*.csv` 기준 최고 val accuracy:
  - `efficientnet_b0`: epoch 3에서 `0.9168`
  - `mobilenet_v2`: epoch 4에서 `0.9140`
  - `resnet50`: epoch 5에서 `0.8909`
  - `simplecnn`: epoch 5에서 `0.0490`
- 학습 경향 요약:
  - Transfer learning 모델은 epoch 2~4 구간에서 빠르게 수렴했습니다.
  - `simplecnn`은 train/val 모두 저정확도 구간에 머물러 underfitting 경향을 보였습니다.
- 제출 시 포함 권장 그래프:
  - epoch vs train_loss, val_loss
  - epoch vs train_acc, val_acc

## 10) GUI Demo

Streamlit 실행:

```bash
streamlit run streamlit_app.py
```

### Demo Case A (노란 피규어 이미지)

- 입력 이미지: 피카츄와 유사한 노란 피규어
- 모델별 예측:
  - `resnet50`: `Abra` (67.50%)
  - `mobilenet_v2`: `Hypno` (36.63%)
  - `efficientnet_b0`: `Jolteon` (51.50%)
  - `simplecnn`: `Pikachu` (2.84%)
- 해석: 스타일화된 피규어 형태에서는 모델 간 예측이 갈렸고, 신뢰도도 상대적으로 낮았습니다.

![GUI Demo Case A](./assets/gui_demo_case_a.png)

### Demo Case B (피카츄 이미지)

- 입력 이미지: 피카츄(모자 착용) 이미지
- 모델별 예측:
  - `resnet50`: `Pikachu` (100.00%)
  - `mobilenet_v2`: `Pikachu` (96.84%)
  - `efficientnet_b0`: `Pikachu` (99.74%)
  - `simplecnn`: `Pikachu` (7.88%)
- 해석: transfer learning 모델 3개는 높은 확률로 일치된 정답을 예측했습니다.

![GUI Demo Case B](./assets/gui_demo_case_b.png)

> 스크린샷 파일은 프로젝트 기준 `assets/gui_demo_case_a.png`, `assets/gui_demo_case_b.png` 경로에 두면 README에서 바로 표시됩니다.

## 11) Notes

- GPU가 없으면 CPU로 동작하며 학습 시간이 길어질 수 있습니다.
- 최초 실행 시 pretrained weights 다운로드로 시간이 추가 소요될 수 있습니다.


# 2025 전력사용량 예측 (DACON) — Electricity\_Consumption

## 📖 소개 (Introduction)

본 프로젝트는 건물 정보와 기상 데이터를 활용하여 **시간 단위 전력사용량을 예측**하는 회귀 문제입니다. [DACON의 2025 전력사용량 예측 AI 경진대회](https://dacon.io/competitions/official/236531/overview/description)에 참가하여 얻은 결과물입니다.

  - **핵심 목표**: 시공간 정보(기상, 건물, 시간)를 바탕으로 전력사용량(kWh) 예측
  - **평가 지표**: **SMAPE** (Symmetric Mean Absolute Percentage Error)
  - **대회 기간**: 2025-07-14 \~ 2025-09-15

-----

## 🏆 최종 결과 (Results)

  - **Private Leaderboard SMAPE**: **8.19266**
  - **내부 검증 SMAPE (Random Forest)**: **5.3089**
  - **테스트 데이터 SMAPE (Random Forest)**: **12.2516**

-----

## 🧠 주요 아이디어 및 특징 (Key Ideas & Features)

### 1\. 피처 엔지니어링 (Feature Engineering)

  - **시간 주기성 인코딩**: 전력 소비의 24시간 주기성을 모델링하기 위해 삼각함수를 사용했습니다.
      - `cos(t-14.45)`, `sin(t-14.45)`: 0시와 24시의 연속성을 보장하고, 전력 피크 시간(14.45시, 오후 2시 27분)을 기준으로 위상을 조정하여 모델이 시간대별 패턴을 효과적으로 학습하도록 설계했습니다.
  - **계절 추세 근사**: 여름철 전력 사용량 피크를 비선형적으로 모델링하기 위해 2차 함수를 도입했습니다.
      - `(누적일-75)^2`: 6월 1일부터 시작하는 누적일을 기준으로, 8월 14일(누적일 75일)에 최고점에 달하는 포물선 형태의 패턴을 학습시켜 단기 계절성을 정교하게 반영했습니다.
  - **파생 변수 생성**:
      - **냉방률**: `냉방면적 / 연면적`으로, 건물의 실질적인 냉방 수요를 반영했습니다.
      - **weather\_PC1**: 기온, 강수, 풍속, 습도 등 다중공선성이 있는 기상 변수들을 PCA를 통해 단일 주성분으로 축약하여 과적합을 방지했습니다.
  - **범주형 변수 인코딩**:
      - **건물 유형**: 10가지 건물 유형을 One-Hot Encoding하여 각 유형의 특성을 독립적으로 학습하도록 했습니다.

### 2\. 모델링 (Modeling)

  - **후보 모델**: Linear Regression, **Random Forest**, XGBoost 세 가지 모델을 비교했습니다.
  - **최종 모델**: 내부 검증 결과, **Random Forest**가 SMAPE 5.3089로 가장 우수한 성능을 보여 최종 모델로 선정했습니다. 비선형적이고 복잡한 패턴을 잘 학습하며, 이상치에 강건한 장점을 확인했습니다.
  - **모델 파이프라인**: `데이터 전처리 → 학습/검증 데이터 분할 → 모델링 → 평가 → 하이퍼파라미터 튜닝 → 최종 모델 선택`의 체계적인 파이프라인을 구축했습니다.

### 3\. 특성 중요도 분석 (Feature Importance)

Random Forest 모델 기준, 예측에 가장 큰 영향을 미친 변수는 다음과 같습니다.

1.  **건물유형\_IDC(전화국)**: 24시간 운영되는 서버 장비로 인해 상시 전력 사용량이 매우 높아 가장 중요한 변수로 작용했습니다.
2.  **연면적(m²)**: 건물의 전체 크기를 나타내며, 냉난방 및 조명 등 기본적인 전력 소모량과 직결됩니다.
3.  **냉방률**: 여름철 냉방 수요를 직접적으로 반영하여 계절적 변동성을 포착하는 데 핵심적인 역할을 했습니다.
4.  **건물유형\_병원**: 24시간 의료 장비와 조명 운영으로 인해 전력 사용량이 일정하게 높아 중요한 변수로 작용했습니다.

-----

## 📁 레포지토리 구조 및 실행 방법

### 구조

```
data/
  train.csv
  test.csv
  building_info.csv
src/
  ...
phase_analysis.ipynb
모델링평가.ipynb
상관분석.ipynb
예측값저장.ipynb
특성_중요도.ipynb
submission.csv
```

### 환경 설정

```bash
# 1. 가상환경 생성 및 활성화
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2. 필수 패키지 설치
pip install scikit-learn xgboost lightgbm pandas numpy matplotlib
```

### 실행 가이드

본 프로젝트는 **기능별 Jupyter Notebook** 중심으로 구성되어 있습니다. 아래 노트북들을 순차적 또는 필요에 따라 실행하여 전체 분석 과정을 재현할 수 있습니다.

1.  **`상관분석.ipynb`**: 데이터의 기본적인 상관관계를 탐색합니다.
2.  **`phase_analysis.ipynb`**: 시간 변수와 주기성 인코딩의 효과를 분석합니다.
3.  **`모델링평가.ipynb`**: 주요 모델들의 성능을 비교하고 최적 모델을 선정합니다.
4.  **`특성_중요도.ipynb`**: 최종 모델의 특성 중요도를 분석하고 시각화합니다.
5.  **`예측값저장.ipynb`**: 최종 모델을 사용하여 `test.csv`에 대한 예측을 수행하고 `submission.csv`를 생성합니다.

-----

## 🧭 한계 및 개선 방향

  - **데이터 분포 차이**: 검증(Validation) 데이터셋 대비 테스트(Test) 데이터셋에서 성능이 하락했는데 (SMAPE 5.3 -\> 12.2), 이는 두 데이터셋 간의 분포 차이 때문일 가능성이 있습니다.
  - **모델 앙상블 강화**: LightGBM, CatBoost 등 다른 Boosting 계열 모델을 추가하여 앙상블하면 예측 성능과 안정성을 더욱 향상시킬 수 있습니다.
  - **시계열 특화 모델 적용**: LSTM, GRU와 같은 딥러닝 기반의 시계열 모델을 적용하여 시간 의존적인 복잡한 패턴을 학습하는 방안을 고려할 수 있습니다.
  - **피처 엔지니어링 강화**: 특정 시간대와 건물 유형 간의 상호작용(Interaction) 피처를 생성하는 등 더욱 정교한 피처 엔지니어링을 통해 모델의 예측력을 높일 수 있습니다.

-----

## 🔗 관련 링크

  - **대회 페이지**: [2025 전력사용량 예측 AI 경진대회 (DACON)](https://dacon.io/competitions/official/236531/overview/description)
  - **발표 자료**: [AICA 6기 발표 슬라이드](https://docs.google.com/presentation/d/1E8BDLb-l71Km8BVwPziSpypnOvHVZTE3/edit?pli=1&slide=id.p1#slide=id.p1)
  - **GitHub 레포지토리**: [Electricity\_Consumption](https://github.com/CONG878/Electricity_Consumption)
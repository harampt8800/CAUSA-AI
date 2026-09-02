# CAUSA — Causal Audit for Unbiased Scoring & Allocation

금융 AI(신용평가·상품추천 모델)의 집단 간 결과 격차를, 인과매개분석으로 분해하여
"정당한 요인으로 설명되는 부분"과 "설명되지 않는 부분(추가 조사 필요)"을 구분해주는
공정성 스크리닝 도구입니다.

## 폴더 구조

```
causa/
├── backend/          # FastAPI 백엔드
│   ├── app.py
│   ├── aipw_module.py
│   ├── requirements.txt
│   ├── fonts/        # PDF 리포트용 한글 폰트
│   └── synthetic_validation.py   # 방법론 검증 스크립트
├── frontend/         # 프론트엔드 (정적 HTML)
│   └── index.html
└── data/             # 데모/시연용 데이터 (실제 데이터 아님, 합성 생성)
    └── causa_demo_synthetic.csv
```

## 로컬 실행 방법

### 백엔드
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --port 8001
```
(윈도우는 python3 대신 python 사용)

### 프론트엔드
`frontend/index.html` 파일을 더블클릭해서 브라우저로 열면 됩니다.
화면 상단 "백엔드 주소"가 `http://localhost:8001`로 되어 있는지 확인하세요.

### 테스트
`data/causa_demo_synthetic.csv` 파일을 ①번(데이터 업로드) 단계에서 업로드해서
전체 흐름(업로드 → 인과구조 설정 → 효과분해 → What-if → PDF 리포트)을 테스트할 수 있습니다.

## 데이터 안내 (중요)

`data/causa_demo_synthetic.csv`는 실제 데이터가 아니라, 탐색적 분석에서 관찰된 패턴과
유사하게 코드로 직접 생성한 합성 데이터입니다. 실제 원본 데이터셋(AI Hub 등)은 이용약관상
배포·재배포하지 않으며, 실제 데이터에서 관찰된 결과는 기획서/발표자료에 "탐색적 분석 결과"로만
인용합니다.

## 배포

- 백엔드: Render (Web Service, `backend` 폴더를 Root Directory로 지정)
- 프론트엔드: Render Static Site 또는 Vercel (`frontend` 폴더 사용)

## 방법론 검증

`backend/synthetic_validation.py`를 실행하면, 정답을 아는 3종 시나리오(정상군/간접경로군/직접경로군)로
분석 엔진의 정확도를 재확인할 수 있습니다.
```bash
cd backend
python synthetic_validation.py
```

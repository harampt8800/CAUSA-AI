"""
CAUSA 백엔드
- POST /upload          : CSV 업로드 -> 데이터 검증(결측치, 표본수, 희귀집단 여부)
- GET  /dag/template     : 표준 DAG 템플릿 제안 (컬럼 목록 기반 추천)
- POST /dag/approve      : 사용자가 확정한 DAG(보호속성/매개변수/결과변수) 저장 + 승인이력 기록
- POST /analyze          : 승인된 DAG로 총효과/직접효과/간접효과 계산 (회귀 기반 매개분석)
- POST /whatif           : 특정 매개변수를 제외했을 때 공정성-예측력 트레이드오프 비교
- GET  /report/{session_id} : 분석 결과를 PDF 감사리포트로 생성
"""

import io
import os
import uuid
import time
import json
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from aipw_module import diagnose_and_recommend, estimate_aipw

# ---------- 한글 폰트 등록 (시스템 폰트에 의존하지 않고, 함께 배포되는 폰트 파일을 직접 사용) ----------
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
pdfmetrics.registerFont(TTFont("NotoSansKR", os.path.join(FONT_DIR, "NotoSansKR-Regular.ttf")))
pdfmetrics.registerFont(TTFont("NotoSansKR-Bold", os.path.join(FONT_DIR, "NotoSansKR-Bold.ttf")))

app = FastAPI(title="CAUSA API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- 세션 저장소 (프로토타입용 인메모리) ----------
SESSIONS = {}

REPORT_DIR = "/home/claude/causa_backend/reports"
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 1) 데이터 업로드 및 검증
# ============================================================
@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "CSV 파일을 읽을 수 없습니다.")

    session_id = str(uuid.uuid4())[:8]
    SESSIONS[session_id] = {"df": df, "created_at": time.time(), "approval_log": []}

    # 데이터 검증 요약
    missing = df.isnull().sum().to_dict()
    dtypes = df.dtypes.astype(str).to_dict()

    # 범주형 컬럼별 표본 수 (희귀 집단 탐지: 표본 20 미만이면 경고)
    categorical_summary = {}
    for col in df.select_dtypes(include=["object"]).columns:
        counts = df[col].value_counts().to_dict()
        rare = [k for k, v in counts.items() if v < 20]
        categorical_summary[col] = {"value_counts": counts, "rare_categories": rare}

    return {
        "session_id": session_id,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "missing_values": missing,
        "dtypes": dtypes,
        "categorical_summary": categorical_summary,
    }


# ============================================================
# 2) 표준 DAG 템플릿 제안 (컬럼명 기반 휴리스틱 추천, 사용자가 반드시 수정/승인해야 함)
# ============================================================
PROTECTED_KEYWORDS = ["gender", "성별", "age", "연령", "employment", "고용", "occupation", "직업", "disability", "장애"]
MEDIATOR_KEYWORDS = ["income", "소득", "job", "직업", "credit", "신용", "debt", "부채", "asset", "자산"]
OUTCOME_KEYWORDS = ["amt", "금액", "score", "점수", "approved", "승인", "amount"]


@app.get("/dag/template")
async def dag_template(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "세션을 찾을 수 없습니다. 먼저 데이터를 업로드하세요.")
    df = SESSIONS[session_id]["df"]
    cols = list(df.columns)

    def guess(keywords):
        return [c for c in cols if any(k.lower() in c.lower() for k in keywords)]

    # 각 컬럼이 "결과변수로 쓸 수 있는지"(숫자형), "보호속성으로 쓸 수 있는지"(범주 정확히 2개)를
    # 미리 판정해서 프론트엔드가 애초에 부적합한 컬럼을 선택 못 하게(비활성화) 만들 수 있도록 정보를 제공한다.
    column_eligibility = {}
    for c in cols:
        is_numeric = pd.api.types.is_numeric_dtype(df[c])
        n_unique = df[c].nunique(dropna=True)
        column_eligibility[c] = {
            "is_numeric": bool(is_numeric),
            "n_unique_categories": int(n_unique),
            "eligible_as_outcome": bool(is_numeric),
            "eligible_as_protected": bool(n_unique == 2),
        }

    suggested = {
        "protected_candidates": guess(PROTECTED_KEYWORDS),
        "mediator_candidates": guess(MEDIATOR_KEYWORDS),
        "outcome_candidates": guess(OUTCOME_KEYWORDS),
        "all_columns": cols,
        "column_eligibility": column_eligibility,
    }
    return {
        "session_id": session_id,
        "note": "이 추천은 컬럼명 기반의 초안일 뿐이며, 반드시 사용자가 검토·수정 후 승인해야 분석이 실행됩니다.",
        "suggested": suggested,
    }


# ============================================================
# 3) DAG 승인 (사용자 확정)
# ============================================================
class DagApproval(BaseModel):
    session_id: str
    protected_attr: str
    mediators: List[str]
    outcome: str
    approver_note: Optional[str] = ""


@app.post("/dag/approve")
async def approve_dag(payload: DagApproval):
    if payload.session_id not in SESSIONS:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    session = SESSIONS[payload.session_id]
    df = session["df"]

    for col in [payload.protected_attr, payload.outcome] + payload.mediators:
        if col not in df.columns:
            raise HTTPException(400, f"컬럼 '{col}'이 데이터에 없습니다.")

    dag = {
        "protected_attr": payload.protected_attr,
        "mediators": payload.mediators,
        "outcome": payload.outcome,
        "approved_at": time.time(),
        "approver_note": payload.approver_note,
    }
    session["approved_dag"] = dag
    session["approval_log"].append(dag)

    return {"approved": True, "dag": dag}


# ============================================================
# 4) 효과 분해 실행 (회귀 기반 인과매개분석 - 합성데이터로 사전 검증된 방식)
# ============================================================
def run_mediation_analysis(df: pd.DataFrame, protected_attr: str, mediators: List[str], outcome: str):
    work = df.dropna(subset=[protected_attr, outcome] + mediators).copy()
    if len(work) < 30:
        raise HTTPException(400, "분석 가능한 표본이 너무 적습니다 (결측치 제외 후 30건 미만).")

    # 보호속성 이진화 (범주형이면 최빈값이 아닌 값을 1로, 이미 숫자면 그대로)
    # dtype이 'object'뿐 아니라 최신 pandas의 'string' 확장타입일 수도 있어, is_numeric_dtype으로 안전하게 판별
    if not pd.api.types.is_numeric_dtype(work[protected_attr]):
        categories = sorted(work[protected_attr].astype(str).unique())
        if len(categories) != 2:
            raise HTTPException(400, "현재 MVP는 보호속성이 이진 범주(예: 남/여)인 경우만 지원합니다.")
        base_cat, target_cat = categories
        work["_A"] = (work[protected_attr].astype(str) == target_cat).astype(int)
        protected_label = f"{target_cat} (기준집단: {base_cat})"
    else:
        work["_A"] = work[protected_attr].astype(float)
        protected_label = protected_attr

    # 결과변수: 로그변환 (금액류 데이터의 왜도 보정, 0 이상 가정)
    if pd.api.types.is_numeric_dtype(work[outcome]):
        work["_Y"] = np.log1p(work[outcome].clip(lower=0))
        outcome_is_log = True
    else:
        raise HTTPException(400, "결과변수는 숫자형이어야 합니다.")

    # 매개변수 인코딩 (범주형은 원핫, 숫자형은 그대로 표준화 없이 사용)
    mediator_df = pd.get_dummies(work[mediators], drop_first=True) if mediators else pd.DataFrame(index=work.index)
    # 상수 컬럼(단일값) 제거 - 다중공선성 방지
    mediator_df = mediator_df.loc[:, mediator_df.nunique() > 1]

    # 총효과
    X_total = sm.add_constant(pd.DataFrame({"A": work["_A"]}))
    model_total = sm.OLS(work["_Y"], X_total.astype(float)).fit()
    total_effect = model_total.params["A"]

    # 직접효과 (매개변수 통제)
    X_direct = pd.concat([mediator_df, pd.DataFrame({"A": work["_A"]}, index=work.index)], axis=1)
    X_direct = sm.add_constant(X_direct.astype(float))
    model_direct = sm.OLS(work["_Y"], X_direct).fit()
    direct_effect = model_direct.params["A"]
    direct_p = model_direct.pvalues["A"]
    ci = model_direct.conf_int().loc["A"].values

    indirect_effect = total_effect - direct_effect

    def to_pct(x):
        return float((np.exp(x) - 1) * 100)

    # ---------- 데이터 특성 자동진단 + AIPW 교차검증 ----------
    # 표본크기, 그룹간 데이터 겹침(overlap)을 보고 회귀 vs AIPW 중 어느 쪽이 더 신뢰할 만한지 진단하고,
    # 두 방법 모두 실제로 계산해서 서로 비슷한 결론을 내는지(강건성) 교차검증한다.
    method_diagnostics = None
    aipw_result = None
    try:
        diag = diagnose_and_recommend(work, work["_A"], mediator_df)
        aipw_result = estimate_aipw(work, work["_A"], mediator_df, work["_Y"], propensity=diag["propensity_scores"])
        method_diagnostics = {
            "n_samples": diag["n_samples"],
            "extreme_propensity_ratio": diag["extreme_propensity_ratio"],
            "recommended_method": diag["recommended_method"],
            "reasons": diag["reasons"],
        }
    except Exception as e:
        method_diagnostics = {"error": f"AIPW 교차검증을 계산하지 못했습니다: {str(e)}"}

    cross_check_agree = None
    if aipw_result is not None:
        # 두 방법의 방향(부호)이 같고, 크기 차이가 과도하게 크지 않은지로 "강건성 일치 여부" 판단
        same_sign = (direct_effect * aipw_result["aipw_effect_log"]) >= 0
        magnitude_ratio = abs(aipw_result["aipw_effect_log"] - direct_effect) / (abs(direct_effect) + 1e-6)
        cross_check_agree = bool(same_sign and magnitude_ratio < 0.5)

    # ---------- 쉬운 말 요약 (금융권 비전문가도 바로 이해할 수 있도록) ----------
    direct_pct_val = to_pct(direct_effect)
    higher_or_lower = "높습니다" if direct_pct_val > 0 else "낮습니다"
    magnitude_word = "뚜렷하게" if abs(direct_pct_val) >= 10 else ("눈에 띄게" if abs(direct_pct_val) >= 5 else "약간")
    if direct_p < 0.05:
        action_word = "추가로 자세히 조사해볼 필요가 있습니다."
    else:
        action_word = "통계적으로 우연히 나타났을 가능성이 있어, 지금 당장 조치가 필요한 수준은 아닙니다."

    plain_summary = (
        f"쉽게 말하면: '{protected_label}' 쪽이 비교 대상보다 결과값이 평균적으로 {magnitude_word} "
        f"{abs(direct_pct_val):.1f}% {higher_or_lower}. 이 차이는 소득·직업 같이 이미 인정한 정당한 이유들로는 "
        f"다 설명되지 않는 부분이며, {action_word}"
    )

    return {
        "protected_label": protected_label,
        "n_samples": len(work),
        "outcome_transform": "log1p" if outcome_is_log else "none",
        "total_effect_log": float(total_effect),
        "direct_effect_log": float(direct_effect),
        "indirect_effect_log": float(indirect_effect),
        "total_effect_pct": to_pct(total_effect),
        "direct_effect_pct": to_pct(direct_effect),
        "indirect_effect_pct": to_pct(indirect_effect),
        "direct_effect_p_value": float(direct_p),
        "direct_effect_ci_95": [float(ci[0]), float(ci[1])],
        "direct_effect_significant": bool(direct_p < 0.05),
        "model_r_squared": float(model_direct.rsquared),
        "high_risk_signal": bool(direct_p < 0.05),
        "interpretation": (
            "직접효과가 통계적으로 유의합니다. 매개변수(정당한 요인)로 설명되지 않는 격차가 존재하여 "
            "추가 조사가 필요한 고위험 신호로 표시됩니다. 이는 차별의 최종 판정이 아니라 스크리닝 결과입니다."
            if direct_p < 0.05 else
            "직접효과가 통계적으로 유의하지 않습니다. 관찰된 격차는 지정된 매개변수로 상당 부분 설명 가능합니다."
        ),
        "plain_summary": plain_summary,
        "method_diagnostics": method_diagnostics,
        "aipw_result": aipw_result,
        "cross_check_agree": cross_check_agree,
    }


class AnalyzeRequest(BaseModel):
    session_id: str


@app.post("/analyze")
async def analyze(payload: AnalyzeRequest):
    if payload.session_id not in SESSIONS:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    session = SESSIONS[payload.session_id]
    if "approved_dag" not in session:
        raise HTTPException(400, "DAG가 아직 승인되지 않았습니다. /dag/approve를 먼저 호출하세요.")

    dag = session["approved_dag"]
    result = run_mediation_analysis(session["df"], dag["protected_attr"], dag["mediators"], dag["outcome"])
    session["analysis_result"] = result
    return result


# ============================================================
# 5) What-if 비교: 특정 매개변수를 제거했을 때 공정성-예측력 트레이드오프
# ============================================================
class WhatIfRequest(BaseModel):
    session_id: str
    exclude_mediator: str  # 제외해볼 매개변수 하나


@app.post("/whatif")
async def whatif(payload: WhatIfRequest):
    if payload.session_id not in SESSIONS:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    session = SESSIONS[payload.session_id]
    if "approved_dag" not in session:
        raise HTTPException(400, "DAG가 먼저 승인되어야 합니다.")

    dag = session["approved_dag"]
    df = session["df"]

    baseline = run_mediation_analysis(df, dag["protected_attr"], dag["mediators"], dag["outcome"])

    remaining_mediators = [m for m in dag["mediators"] if m != payload.exclude_mediator]
    scenario = run_mediation_analysis(df, dag["protected_attr"], remaining_mediators, dag["outcome"])

    r2_drop = baseline['model_r_squared'] - scenario['model_r_squared']
    if r2_drop > 0.1:
        importance_word = "이 요인은 결과를 예측하는 데 매우 중요한 역할을 하고 있었습니다."
    elif r2_drop > 0.02:
        importance_word = "이 요인은 결과 예측에 어느 정도 영향을 주고 있었습니다."
    else:
        importance_word = "이 요인을 빼도 예측력이 거의 변하지 않아, 실제로는 큰 영향을 주지 않았던 것으로 보입니다."

    effect_change = abs(scenario['direct_effect_pct'] - baseline['direct_effect_pct'])
    if effect_change < 1:
        fairness_word = "격차(직접효과)는 거의 변하지 않았습니다."
    else:
        fairness_word = f"격차(직접효과)가 {effect_change:.1f}%p 정도 변했습니다."

    plain_whatif_summary = (
        f"쉽게 말하면: '{payload.exclude_mediator}'라는 기준을 빼고 다시 계산해봤습니다. {importance_word} "
        f"{fairness_word} 즉, 이 기준을 정당한 이유로 인정할지 말지에 따라 결과 해석이 달라질 수 있다는 뜻입니다."
    )

    return {
        "baseline": {
            "mediators_used": dag["mediators"],
            "direct_effect_pct": baseline["direct_effect_pct"],
            "model_r_squared": baseline["model_r_squared"],
        },
        "scenario_excluding": {
            "mediators_used": remaining_mediators,
            "excluded": payload.exclude_mediator,
            "direct_effect_pct": scenario["direct_effect_pct"],
            "model_r_squared": scenario["model_r_squared"],
        },
        "tradeoff_note": (
            f"'{payload.exclude_mediator}'을(를) 매개변수에서 제외하면 직접효과가 "
            f"{baseline['direct_effect_pct']:+.1f}% -> {scenario['direct_effect_pct']:+.1f}% 로 변화하고, "
            f"모델 설명력(R²)은 {baseline['model_r_squared']:.3f} -> {scenario['model_r_squared']:.3f} 로 변화합니다. "
            "이는 공정성 지표와 예측력 사이의 트레이드오프를 보여주는 참고 자료이며, 정책 결정은 담당자가 수행해야 합니다."
        ),
        "plain_whatif_summary": plain_whatif_summary,
    }


# ============================================================
# 6) PDF 감사 리포트 생성
# ============================================================
@app.get("/report/{session_id}")
async def generate_report(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
    session = SESSIONS[session_id]
    if "analysis_result" not in session:
        raise HTTPException(400, "분석이 아직 실행되지 않았습니다.")

    result = session["analysis_result"]
    dag = session["approved_dag"]

    filepath = os.path.join(REPORT_DIR, f"{session_id}_report.pdf")
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    styles = getSampleStyleSheet()

    # 기본 스타일들을 한글 폰트로 교체 (Title, Heading2, Normal)
    styles["Title"].fontName = "NotoSansKR-Bold"
    styles["Heading2"].fontName = "NotoSansKR-Bold"
    styles["Normal"].fontName = "NotoSansKR"

    story = []

    story.append(Paragraph("CAUSA 공정성 진단 감사 리포트", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"세션 ID: {session_id}", styles["Normal"]))
    story.append(Paragraph(f"생성 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("1. 분석 설정 (DAG 승인 이력)", styles["Heading2"]))
    dag_table_data = [
        ["항목", "값"],
        ["보호속성", dag["protected_attr"]],
        ["매개변수", ", ".join(dag["mediators"])],
        ["결과변수", dag["outcome"]],
        ["승인 시각", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dag["approved_at"]))],
    ]
    t = Table(dag_table_data, colWidths=[120, 320])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a3348")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, -1), "NotoSansKR"),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    story.append(Paragraph("2. 효과 분해 결과", styles["Heading2"]))
    if result.get("plain_summary"):
        story.append(Paragraph(result["plain_summary"], styles["Normal"]))
        story.append(Spacer(1, 10))
    result_table_data = [
        ["지표", "값"],
        ["표본 수", str(result["n_samples"])],
        ["총효과", f"{result['total_effect_pct']:+.1f}%"],
        ["간접효과 (매개변수 경로)", f"{result['indirect_effect_pct']:+.1f}%"],
        ["직접효과 (설명 불가 구간)", f"{result['direct_effect_pct']:+.1f}%"],
        ["직접효과 p-value", f"{result['direct_effect_p_value']:.3e}" if result['direct_effect_p_value'] < 0.0001 else f"{result['direct_effect_p_value']:.6f}"],
        ["직접효과 95% 신뢰구간(로그)", f"[{result['direct_effect_ci_95'][0]:.4f}, {result['direct_effect_ci_95'][1]:.4f}]"],
        ["고위험 신호 여부", "예 (추가 조사 권고)" if result["high_risk_signal"] else "아니오"],
        ["모델 설명력 (R²)", f"{result['model_r_squared']:.3f}"],
    ]
    t2 = Table(result_table_data, colWidths=[180, 260])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a3348")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, -1), "NotoSansKR"),
    ]))
    story.append(t2)
    story.append(Spacer(1, 20))

    story.append(Paragraph("3. 해석 및 한계", styles["Heading2"]))
    story.append(Paragraph(result["interpretation"], styles["Normal"]))
    story.append(Spacer(1, 10))
    if result["n_samples"] > 10000:
        story.append(Paragraph(
            f"참고: 본 분석의 표본 수는 {result['n_samples']}건으로 매우 큽니다. 표본이 클수록 아주 작은 격차도 "
            "통계적으로 유의(p-value가 매우 작게)하게 나타날 수 있으므로, p-value 자체보다 효과크기(직접효과 %)의 "
            "실질적 크기를 함께 살펴보는 것이 중요합니다.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 10))
    story.append(Paragraph(
        "본 분석은 명시된 DAG(인과 그래프) 가정에 기반한 스크리닝 결과이며, 완벽한 인과관계를 증명하는 것이 아닙니다. "
        "최종적인 차별 여부와 후속 조치는 반드시 컴플라이언스 담당자와 도메인 전문가의 검토를 거쳐 결정되어야 합니다.",
        styles["Normal"]
    ))

    doc.build(story)

    return FileResponse(filepath, media_type="application/pdf", filename="CAUSA_감사리포트.pdf")


@app.get("/")
async def root():
    return {"status": "CAUSA API running"}

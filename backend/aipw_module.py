"""
AIPW(이중강건추정) 모듈 + 데이터 특성 자동진단 로직
- 회귀분석과 AIPW 두 방법으로 각각 직접효과를 추정하고 교차검증
- 표본크기, 그룹간 데이터 겹침(positivity/overlap)을 보고 어떤 방법이 더 신뢰할 만한지 추천
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def diagnose_and_recommend(df, protected_col_binary, mediator_dummies):
    """
    데이터 특성을 진단해서 회귀 vs AIPW 중 어느 쪽을 더 신뢰할지 추천한다.
    - 표본 크기
    - 그룹간 겹침(overlap/positivity): 매개변수만으로 보호속성을 얼마나 잘 예측해버리는지
      -> 너무 잘 예측되면(=겹치는 표본이 적으면) 두 방법 다 불안정해지지만, 특히 회귀가 더 위험해짐
    """
    n = len(df)
    X = sm.add_constant(mediator_dummies.astype(float))
    y = protected_col_binary.astype(float)

    try:
        logit_model = sm.Logit(y, X).fit(disp=0)
        propensity = logit_model.predict(X)
    except Exception:
        # 완전분리(perfect separation) 등으로 로지스틱 회귀가 실패하는 경우
        propensity = pd.Series(np.clip(y.mean(), 0.05, 0.95), index=df.index)

    # overlap 진단: 성향점수가 0.05 미만 또는 0.95 초과인 비율
    extreme_ratio = float(((propensity < 0.05) | (propensity > 0.95)).mean())

    reasons = []
    if n < 3000:
        recommended = "regression"
        reasons.append(f"표본 수({n}건)가 적어 AIPW의 두 모델(성향점수+결과예측)이 불안정할 수 있습니다.")
    elif extreme_ratio > 0.15:
        recommended = "regression"
        reasons.append(f"매개변수만으로 보호속성이 강하게 예측되는 표본이 {extreme_ratio*100:.1f}%로 많아(겹침 부족), AIPW 가중치가 불안정해질 수 있습니다.")
    else:
        recommended = "aipw"
        reasons.append(f"표본 수({n}건)가 충분하고 그룹간 데이터 겹침도 양호하여(극단치 비율 {extreme_ratio*100:.1f}%), 이중강건추정(AIPW)의 장점을 안정적으로 활용할 수 있습니다.")

    return {
        "n_samples": n,
        "extreme_propensity_ratio": extreme_ratio,
        "recommended_method": recommended,
        "reasons": reasons,
        "propensity_scores": propensity,
    }


def estimate_aipw(df, protected_col_binary, mediator_dummies, outcome_log, propensity=None):
    """
    AIPW(Augmented Inverse Propensity Weighting) 추정.
    - 성향점수모델: 매개변수 -> 보호속성 예측 (로지스틱회귀)
    - 결과모델: 매개변수 -> 결과 예측 (그룹별 선형회귀, T-learner 방식)
    - 두 모델을 결합해 이중강건추정치 산출
    """
    n = len(df)
    A = protected_col_binary.values.astype(float)
    Y = outcome_log.values.astype(float)
    X = sm.add_constant(mediator_dummies.astype(float))

    if propensity is None:
        try:
            logit_model = sm.Logit(A, X).fit(disp=0)
            e = logit_model.predict(X).values
        except Exception:
            e = np.full(n, A.mean())
    else:
        e = propensity.values

    e = np.clip(e, 0.02, 0.98)  # 극단값 클리핑으로 안정성 확보

    # 결과모델: 그룹별로 따로 학습 (T-learner)
    idx1 = A == 1
    idx0 = A == 0
    model1 = sm.OLS(Y[idx1], X[idx1]).fit()
    model0 = sm.OLS(Y[idx0], X[idx0]).fit()
    mu1 = model1.predict(X)
    mu0 = model0.predict(X)

    aipw_terms = (A * (Y - mu1) / e + mu1) - ((1 - A) * (Y - mu0) / (1 - e) + mu0)
    ate_aipw = float(np.mean(aipw_terms))
    se_aipw = float(np.std(aipw_terms, ddof=1) / np.sqrt(n))
    ci = (ate_aipw - 1.96 * se_aipw, ate_aipw + 1.96 * se_aipw)

    return {
        "aipw_effect_log": ate_aipw,
        "aipw_effect_pct": float((np.exp(ate_aipw) - 1) * 100),
        "aipw_se": se_aipw,
        "aipw_ci_95": [float(ci[0]), float(ci[1])],
    }

"""
CAUSA 방법론 검증: 정답을 아는 합성데이터 3종으로 인과매개분석 파이프라인의 정확도를 검증한다.

- 정상군: 보호속성(A)이 결과(Y)에 어떤 경로로도 영향 없음 -> 직접효과, 간접효과 모두 0에 가까워야 함
- 간접경로군: A -> M(매개변수) -> Y 경로만 존재 -> 간접효과는 검출, 직접효과는 0에 가까워야 함
- 직접경로군: 간접경로 + A -> Y 직접경로까지 존재 -> 직접효과가 유의하게 검출되어야 함
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)
N = 5000

def run_mediation(A, M, Y):
    """총효과/직접효과/간접효과 추정 (선형회귀 기반, 우리가 실제 서비스에서 쓰는 방식과 동일)"""
    # 총효과: M을 통제하지 않고 A만으로 Y를 설명
    X_total = sm.add_constant(pd.DataFrame({'A': A}))
    model_total = sm.OLS(Y, X_total).fit()
    total_effect = model_total.params['A']

    # 직접효과: M을 통제한 상태에서 A의 계수
    X_direct = sm.add_constant(pd.DataFrame({'A': A, 'M': M}))
    model_direct = sm.OLS(Y, X_direct).fit()
    direct_effect = model_direct.params['A']
    direct_p = model_direct.pvalues['A']
    direct_ci = model_direct.conf_int().loc['A'].values  # 95% CI

    indirect_effect = total_effect - direct_effect
    return {
        'total': total_effect, 'direct': direct_effect, 'direct_p': direct_p,
        'direct_ci_low': direct_ci[0], 'direct_ci_high': direct_ci[1],
        'indirect': indirect_effect
    }

results = []

# ===== 1) 정상군: A -> M 없음, A -> Y 없음 (M은 A와 무관하게 생성) =====
A1 = np.random.binomial(1, 0.5, N)
M1 = np.random.normal(50, 10, N)  # A와 무관
Y1 = 2.0 * M1 + np.random.normal(0, 5, N)  # A 영향 전혀 없음
true_indirect_1 = 0.0
true_direct_1 = 0.0
res1 = run_mediation(A1, M1, Y1)
results.append(('정상군', true_direct_1, true_indirect_1, res1))

# ===== 2) 간접경로군: A -> M -> Y만 존재 (A -> Y 직접경로 없음) =====
A2 = np.random.binomial(1, 0.5, N)
delta = 8.0  # A가 M에 미치는 영향
beta = 2.0   # M이 Y에 미치는 영향
M2 = 50 + delta * A2 + np.random.normal(0, 10, N)
Y2 = beta * M2 + np.random.normal(0, 5, N)  # A의 직접 영향 없음(계수 0)
true_indirect_2 = delta * beta  # = 16.0
true_direct_2 = 0.0
res2 = run_mediation(A2, M2, Y2)
results.append(('간접경로군', true_direct_2, true_indirect_2, res2))

# ===== 3) 직접경로군: 간접경로 + A -> Y 직접경로 추가 =====
A3 = np.random.binomial(1, 0.5, N)
gamma = 10.0  # A가 Y에 미치는 직접 영향 (진짜 심어놓은 값)
M3 = 50 + delta * A3 + np.random.normal(0, 10, N)
Y3 = beta * M3 + gamma * A3 + np.random.normal(0, 5, N)
true_indirect_3 = delta * beta  # = 16.0
true_direct_3 = gamma           # = 10.0
res3 = run_mediation(A3, M3, Y3)
results.append(('직접경로군', true_direct_3, true_indirect_3, res3))

print(f"{'그룹':10s} {'실제직접':>8s} {'추정직접':>8s} {'오차':>6s} {'CI포함':>6s} {'p<0.05검출':>10s} | {'실제간접':>8s} {'추정간접':>8s} {'오차':>6s}")
for name, true_direct, true_indirect, res in results:
    direct_err = abs(res['direct'] - true_direct)
    ci_covers = res['direct_ci_low'] <= true_direct <= res['direct_ci_high']
    detected = res['direct_p'] < 0.05
    indirect_err = abs(res['indirect'] - true_indirect)
    print(f"{name:10s} {true_direct:>8.2f} {res['direct']:>8.2f} {direct_err:>6.2f} {str(ci_covers):>6s} {str(detected):>10s} | {true_indirect:>8.2f} {res['indirect']:>8.2f} {indirect_err:>6.2f}")

print()
print("=== 기대 결과와 비교 ===")
print("정상군: 직접효과 미검출 기대 ->", "통과" if not results[0][3]['direct_p']<0.05 else "실패")
print("간접경로군: 직접효과 미검출 기대 ->", "통과" if not results[1][3]['direct_p']<0.05 else "실패")
print("직접경로군: 직접효과 검출 기대 ->", "통과" if results[2][3]['direct_p']<0.05 else "실패")

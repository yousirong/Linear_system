import numpy as np
from scipy.integrate import simps, quad, fixed_quad
from scipy.special import roots_legendre

def quad_explain(func, a, b, epsabs=1.49e-8, epsrel=1.49e-8, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=None, limlst=None):
    """
    Compute a definite integral using quad with explanation.

    Parameters:
        func : callable
            A Python function or method to integrate.
        a, b : float
            The limits of integration.
        epsabs, epsrel : float, optional
            The absolute and relative tolerances. Default is 1.49e-8 for both.
        limit : int, optional
            The maximum number of subintervals to use. Default is 50.
        points : array_like, optional
            If given, these points are used as the integration nodes. Default is None.
        weight : array_like, optional
            If given and points is not None, these weights are applied to the integration nodes. Default is None.
        wvar : tuple, optional
            Additional arguments to pass to the weight function.
        wopts : dict, optional
            Additional options to pass to the weight function.
        maxp1 : int, optional
            If given, divide the range into at most maxp1 segments. Default is None.
        limlst : int, optional
            If given, stop subdividing when len(segment) < limlst. Default is None.

    Returns:
        result : float
            The integral of func from a to b.
        explanation : str
            Explanation of the integration process.
    """
# quad 함수가 3개의 값을 반환한다고 가정하면,
    result, detail, _ = quad(func, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit, points=points, weight=weight, wvar=wvar, wopts=wopts, maxp1=maxp1, limlst=limlst, full_output=True)
    # detail이 float 타입인지 확인
    if isinstance(detail, float):
        explanation = str(detail)
    else:
        explanation = detail['message']
    return result, explanation

def flength(x, h):
    return np.sqrt(1 + dfp511(x, h)**2)

def dfp511(x, h):
    return (fp511(x + h) - fp511(x - h)) / (2 * h)

def fp511(x):
    return np.sqrt(np.maximum(1 - x**2, 0))

a, b = -1, 1
N = 1000  # Simpson 방법을 위한 세그먼트 수
tol = 1e-6  # 오차 허용치
M = 20  # Gauss-Legendre 적분을 위한 그리드 포인트 수
IT = np.pi  # 실제 적분 값
h_values = [1e-3, 1e-4, 1e-5]  # 수치 미분을 위한 단계 크기

results = np.zeros((len(h_values), 5))  # 다양한 방법에 대한 결과 저장: Simpson, 적응형 Simpson, quad, quadl, Gauss-Legendre

for i, h in enumerate(h_values):
    # 다양한 방법으로 적분 계산
    Is = simps(flength(np.linspace(a, b, N), h), np.linspace(a, b, N))
    Ias, _ = quad(lambda x: flength(x, h), a, b, epsrel=tol)
    Iq, _ = quad_explain(lambda x: flength(x, h), a, b, epsrel=tol)
    Iql, _ = fixed_quad(lambda x: flength(x, h), a, b, n=N)

    # Gauss-Legendre 적분
    nodes, weights = roots_legendre(M)
    x_nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    IGL = np.sum(weights * flength(x_nodes, h)) * 0.5 * (b - a)

    # 결과 저장
    results[i, :] = [Is, Ias, Iq, Iql, IGL]

# 결과 출력
print('결과:')
print(results)

# 오차 계산
errors = np.abs(results - IT)

# 오차 출력
print('오차:')
print(errors)

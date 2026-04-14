# Auto-generated piecewise classifier — no fitted constants in features
# Features are physically motivated (Goldschmidt t, Bartel tau, etc.)
# Tree thresholds were learned from 576 experimentally characterized ABX3 compounds

def predict_perovskite(rA, rB, rX, nA, nB, nX):
    """Return 1 (perovskite) or -1 (non-perovskite)."""
    import math, numpy as np
    log_ratio = math.log(max(rA / rB, 1e-9))
    t   = (rA + rX) / (math.sqrt(2) * (rB + rX))
    tau = rX/rB - nA * (nA - (rA/rB) / log_ratio)
    mu  = rB / rX
    chi = nA / nB
    sig = rA / rX
    dlt = (nA * rA + nB * rB) / (rA + rB)
    nu  = nA * nB / (nX ** 2)
    rho = rB / rA
    X   = [t, tau, mu, chi, sig, dlt, nu, rho]

    if X[1] <= 4.1844:  # tau <= 4.1844
        if X[1] <= 3.9653:  # tau <= 3.9653
            if X[0] <= 1.1221:  # t <= 1.1221
                return 1  # 0/1 train samples
            else:  # t > 1.1221
                return -1  # 0/1 train samples
        else:  # tau > 3.9653
            if X[4] <= 1.0336:  # sig <= 1.0336
                return 1  # 0/1 train samples
            else:  # sig > 1.0336
                return -1  # 0/1 train samples
    else:  # tau > 4.1844
        if X[2] <= 0.4662:  # mu <= 0.4662
            if X[4] <= 1.3782:  # sig <= 1.3782
                return -1  # 0/1 train samples
            else:  # sig > 1.3782
                return 1  # 1/1 train samples
        else:  # mu > 0.4662
            if X[1] <= 4.4763:  # tau <= 4.4763
                return -1  # 0/1 train samples
            else:  # tau > 4.4763
                return -1  # 0/1 train samples

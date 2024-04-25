import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate


def dw_reco_of_t(t_, with_discounting_, c_baseline_, delta_c_0_, lambda_h_, rho_, eta_):
    loss = loss = delta_c_0_ * np.exp(-lambda_h_ * t_)
    if with_discounting_:
        discount = np.exp(-rho_ * t_)
    else:
        discount = 1
    return 1 / (1 - eta_) * discount * (c_baseline_ ** (1 - eta_) - (c_baseline_ - loss) ** (1 - eta_))


def dw_reco(with_discounting_, c_baseline_, delta_c_0_, lambda_h_, rho_, eta_):
    return integrate.quad(dw_reco_of_t, 0, np.inf, args=(with_discounting_, c_baseline_, delta_c_0_, lambda_h_, rho_, eta_))[0]


def dw_long_term(c_baseline_, eta_, delta_c_0_, lambda_h_):
    return c_baseline_ ** (-eta_) * delta_c_0_ / lambda_h_


if __name__ == '__main__':
    # parameters
    c_baseline = 30000
    delta_c_0 = 60
    lambda_h = 6
    rho = 0.06
    eta = 1.5

    # compute welfare loss
    dw_with_discounting = dw_reco(True, c_baseline, delta_c_0, lambda_h, rho, eta)
    dw_without_discounting = dw_reco(False, c_baseline, delta_c_0, lambda_h, rho, eta)
    dw_full_offset_long_term = dw_long_term(c_baseline, eta, delta_c_0, lambda_h)

    print(f'dw_with_discounting: {dw_with_discounting}, dw_without_discounting: {dw_without_discounting}, '
            f'dw_full_offset_long_term: {dw_full_offset_long_term}')

    # variation of c_baseline
    c_baselines = np.linspace(20000, 40000, 100)
    dw_with_discounting = np.array([dw_reco(True, c, delta_c_0, lambda_h, rho, eta) for c in c_baselines])
    dw_without_discounting = np.array([dw_reco(False, c, delta_c_0, lambda_h, rho, eta) for c in c_baselines])
    dw_full_offset_long_term = np.array([dw_long_term(c, eta, delta_c_0, lambda_h) for c in c_baselines])

    # plot
    fig, ax = plt.subplots()
    plt.plot(c_baselines, dw_with_discounting, label='with discounting')
    plt.plot(c_baselines, dw_without_discounting, label='without discounting')
    plt.plot(c_baselines, dw_full_offset_long_term, label='full offset long term')


    # variation of delta_c_0
    delta_c_0s = np.linspace(0, 2000, 1000)
    dw_with_discounting = np.array([dw_reco(True, c_baseline, d, lambda_h, rho, eta) for d in delta_c_0s])
    dw_without_discounting = np.array([dw_reco(False, c_baseline, d, lambda_h, rho, eta) for d in delta_c_0s])
    dw_full_offset_long_term = np.array([dw_long_term(c_baseline, eta, d, lambda_h) for d in delta_c_0s])

    # plot
    fig, ax = plt.subplots()
    plt.plot(delta_c_0s, dw_with_discounting, label='with discounting')
    plt.plot(delta_c_0s, dw_without_discounting, label='without discounting')
    plt.plot(delta_c_0s, dw_full_offset_long_term, label='full offset long term')


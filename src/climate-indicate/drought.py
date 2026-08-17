import numpy as np
from scipy import stats
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PCP_BASE = Path(r'D:\DATA_STORAGE\_data\_DataReview\dailyClimateData_USTH')

month = range(1, 13)

def spi(precip, scale=3):
    accum = np.convolve(precip, np.ones(scale), 'valid')
    finite = np.isfinite(accum)

    zeros = (accum == 0) & finite
    nonzero = accum[finite & ~zeros]
    q = zeros.sum() / finite.sum()          # chia theo số mẫu hợp lệ

    alpha, loc, beta = stats.gamma.fit(nonzero, floc=0)   # tìm kiếm alpha, loc, beta để khớp CDF

    G = stats.gamma.cdf(accum, a=alpha, scale=beta)
    H = np.where(zeros, q, q + (1 - q) * G)  # áp dụng cho MỌI tháng
    H = np.clip(H, 1e-6, 1 - 1e-6)
    spi = stats.norm.ppf(H)
    return spi




def main():
    pcp_daily = pd.read_csv(PCP_BASE / 'R_BATO.day', parse_dates=['date']).set_index('date').iloc[:, -1]
    pcp_monthly = pcp_daily.resample('ME').sum(min_count=1)  # cần có cột 'date' làm index
    # print(pcp_monthly)

    spi_3 = spi(pcp_monthly.values, scale=3)
    print(spi_3)


if __name__ == '__main__':
    main()
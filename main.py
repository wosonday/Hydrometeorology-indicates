import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from climate_indicate.drought import spi


PCP_BASE = Path(r'D:\DATA_STORAGE\_data\_DataReview\dailyClimateData_USTH')

month = range(1, 13)



def main():
    pcp_daily = pd.read_csv(PCP_BASE / 'R_BATO.day', parse_dates=['date']).set_index('date').iloc[:, -1]
    pcp_monthly = pcp_daily.resample('ME').sum(min_count=1)  # cần có cột 'date' làm index
    # print(pcp_monthly)

    spi_3 = spi(pcp_monthly.values, scale=3)
    print(spi_3)


if __name__ == '__main__':
    main()

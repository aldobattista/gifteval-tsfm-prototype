# Run 5 Best Soup (`+SymAvg+Clamp`) — Top-100 Forecast Showcase Book

Comprehensive English visual compendium of the strongest zero-shot forecast windows from `checkpoints/run5_coupled_synth_mix/model_soup.safetensors` (`16,317,344` parameters, `sym_avg=True` + non-negative quantile clamping on non-negative series).

- **Full Vector Book (108 Pages PDF):** [`Run5_Top100_Forecast_Showcase_Book.pdf`](Run5_Top100_Forecast_Showcase_Book.pdf)
- **Catalog Tables:** [`univariate_top50_catalog.csv`](univariate_top50_catalog.csv) | [`multivariate_top50_catalog.csv`](multivariate_top50_catalog.csv)
- **Reproducible Generator:** [`generate_showcase_book.py`](generate_showcase_book.py)

## Executive Summary

- **Part I — Top 50 Univariate Plates (`univariate_top50/`):** Spans all **6 pure-univariate domains** (`Econ/Fin`, `Energy`, `Healthcare`, `Nature`, `Sales`, `Transport`), median **MAE reduction = `76.7%`** vs. `Seasonal_Naive`, median **horizon `R^2` = `0.891`**, median **`Cov80` = `96%`**.
- **Part II — Top 50 Multivariate & Covariate Ablation Plates (`multivariate_top50/`):** Runs the exact same window **WITH vs. WITHOUT** companion signals (`28/50` plates include `past_feat_dynamic_real` and also show the intermediate **Co-Targets Only** trace when `V_tgt > 1`), achieving median **MAE reduction = `16.8%`** over isolated univariate inference and median **`Cov80` = `98%`**.

![Top-100 Book Overview](overview_charts/book_summary_overview.png)

---

## Part I: Top 50 Zero-Shot Univariate Examples

| # | Dataset Config | Domain | Freq | H | Win # | `MAE` Run 5 | `MAE` SN | Δ MAE | `wQL` | `Cov80` | `R^2` | Plot |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | `electricity/D/short` | Energy | `D` | 30 | 1835 | 552.3 | 7116.2 | **-92.2%** | 0.0406 | 100% | 0.973 | [PNG](univariate_top50/uni_01_electricity_D_short_w1835.png) |
| 02 | `m4_yearly/A/short` | Econ/Fin | `A` | 6 | 9420 | 65.88 | 739.8 | **-91.1%** | 0.0359 | 100% | 0.956 | [PNG](univariate_top50/uni_02_m4_yearly_A_short_w9420.png) |
| 03 | `m4_yearly/A/short` | Econ/Fin | `A` | 6 | 22146 | 162.0 | 1722.8 | **-90.6%** | 0.0308 | 100% | 0.962 | [PNG](univariate_top50/uni_03_m4_yearly_A_short_w22146.png) |
| 04 | `m4_weekly/W/short` | Econ/Fin | `W` | 13 | 242 | 212.2 | 1497.0 | **-85.8%** | 0.0687 | 85% | 0.935 | [PNG](univariate_top50/uni_04_m4_weekly_W_short_w242.png) |
| 05 | `m4_yearly/A/short` | Econ/Fin | `A` | 6 | 2148 | 149.0 | 1419.5 | **-89.5%** | 0.0254 | 100% | 0.947 | [PNG](univariate_top50/uni_05_m4_yearly_A_short_w2148.png) |
| 06 | `electricity/H/short` | Energy | `H` | 48 | 2661 | 7.37 | 48.98 | **-85.0%** | 0.0957 | 92% | 0.954 | [PNG](univariate_top50/uni_06_electricity_H_short_w2661.png) |
| 07 | `m4_yearly/A/short` | Econ/Fin | `A` | 6 | 7272 | 52.67 | 414.1 | **-87.3%** | 0.0233 | 100% | 0.916 | [PNG](univariate_top50/uni_07_m4_yearly_A_short_w7272.png) |
| 08 | `electricity/H/short` | Energy | `H` | 48 | 2555 | 134.4 | 744.0 | **-81.9%** | 0.0848 | 81% | 0.899 | [PNG](univariate_top50/uni_08_electricity_H_short_w2555.png) |
| 09 | `covid_deaths/D/short` | Healthcare | `D` | 30 | 22 | 8.03 | 66.93 | **-88.0%** | 0.0415 | 100% | 0.898 | [PNG](univariate_top50/uni_09_covid_deaths_D_short_w22.png) |
| 10 | `electricity/15T/short` | Energy | `15T` | 48 | 2661 | 2.61 | 15.88 | **-83.6%** | 0.1199 | 92% | 0.892 | [PNG](univariate_top50/uni_10_electricity_15T_short_w2661.png) |
| 11 | `us_births/D/short` | Healthcare | `D` | 30 | 10 | 276.7 | 1982.0 | **-86.0%** | 0.0206 | 80% | 0.895 | [PNG](univariate_top50/uni_11_us_births_D_short_w10.png) |
| 12 | `covid_deaths/D/short` | Healthcare | `D` | 30 | 194 | 14.00 | 111.8 | **-87.5%** | 0.0506 | 100% | 0.891 | [PNG](univariate_top50/uni_12_covid_deaths_D_short_w194.png) |
| 13 | `m4_hourly/H/short` | Econ/Fin | `H` | 48 | 35 | 574.9 | 2148.2 | **-73.2%** | 0.0251 | 100% | 0.956 | [PNG](univariate_top50/uni_13_m4_hourly_H_short_w35.png) |
| 14 | `saugeen/W/short` | Nature | `W` | 8 | 16 | 1.91 | 25.60 | **-92.5%** | 0.1455 | 100% | 0.878 | [PNG](univariate_top50/uni_14_saugeen_W_short_w16.png) |
| 15 | `m4_hourly/H/short` | Econ/Fin | `H` | 48 | 145 | 51.00 | 186.2 | **-72.6%** | 0.1250 | 81% | 0.921 | [PNG](univariate_top50/uni_15_m4_hourly_H_short_w145.png) |
| 16 | `electricity/D/short` | Energy | `D` | 30 | 478 | 82.95 | 497.7 | **-83.3%** | 0.0608 | 80% | 0.848 | [PNG](univariate_top50/uni_16_electricity_D_short_w478.png) |
| 17 | `electricity/15T/medium` | Energy | `15T` | 480 | 2661 | 3.61 | 13.56 | **-73.4%** | 0.1762 | 90% | 0.899 | [PNG](univariate_top50/uni_17_electricity_15T_medium_w2661.png) |
| 18 | `m4_hourly/H/short` | Econ/Fin | `H` | 48 | 29 | 2341.6 | 7191.4 | **-67.4%** | 0.0214 | 100% | 0.972 | [PNG](univariate_top50/uni_18_m4_hourly_H_short_w29.png) |
| 19 | `m4_hourly/H/short` | Econ/Fin | `H` | 48 | 38 | 2373.6 | 7185.4 | **-67.0%** | 0.0215 | 100% | 0.971 | [PNG](univariate_top50/uni_19_m4_hourly_H_short_w38.png) |
| 20 | `loop_seattle/5T/medium` | Transport | `5T` | 480 | 4972 | 1.94 | 8.82 | **-78.0%** | 0.0268 | 76% | 0.851 | [PNG](univariate_top50/uni_20_loop_seattle_5T_medium_w4972.png) |
| 21 | `us_births/D/short` | Healthcare | `D` | 30 | 11 | 283.0 | 1134.3 | **-75.1%** | 0.0203 | 87% | 0.896 | [PNG](univariate_top50/uni_21_us_births_D_short_w11.png) |
| 22 | `electricity/H/long` | Energy | `H` | 720 | 1835 | 67.93 | 303.4 | **-77.6%** | 0.1086 | 82% | 0.876 | [PNG](univariate_top50/uni_22_electricity_H_long_w1835.png) |
| 23 | `electricity/15T/medium` | Energy | `15T` | 480 | 2448 | 6.74 | 26.20 | **-74.3%** | 0.0810 | 89% | 0.903 | [PNG](univariate_top50/uni_23_electricity_15T_medium_w2448.png) |
| 24 | `us_births/D/short` | Healthcare | `D` | 30 | 5 | 275.5 | 932.8 | **-70.5%** | 0.0202 | 83% | 0.910 | [PNG](univariate_top50/uni_24_us_births_D_short_w5.png) |
| 25 | `m4_quarterly/Q/short` | Econ/Fin | `Q` | 8 | 10013 | 21.40 | 146.7 | **-85.4%** | 0.0194 | 100% | 0.802 | [PNG](univariate_top50/uni_25_m4_quarterly_Q_short_w10013.png) |
| 26 | `m4_quarterly/Q/short` | Econ/Fin | `Q` | 8 | 4316 | 36.50 | 237.0 | **-84.6%** | 0.0309 | 100% | 0.826 | [PNG](univariate_top50/uni_26_m4_quarterly_Q_short_w4316.png) |
| 27 | `m4_quarterly/Q/short` | Econ/Fin | `Q` | 8 | 8460 | 189.4 | 1241.5 | **-84.7%** | 0.0196 | 100% | 0.741 | [PNG](univariate_top50/uni_27_m4_quarterly_Q_short_w8460.png) |
| 28 | `covid_deaths/D/short` | Healthcare | `D` | 30 | 135 | 1.07 | 6.77 | **-84.2%** | 0.0390 | 100% | 0.806 | [PNG](univariate_top50/uni_28_covid_deaths_D_short_w135.png) |
| 29 | `covid_deaths/D/short` | Healthcare | `D` | 30 | 106 | 75.90 | 451.6 | **-83.2%** | 0.0324 | 100% | 0.820 | [PNG](univariate_top50/uni_29_covid_deaths_D_short_w106.png) |
| 30 | `us_births/D/short` | Healthcare | `D` | 30 | 3 | 404.0 | 2158.7 | **-81.3%** | 0.0291 | 73% | 0.795 | [PNG](univariate_top50/uni_30_us_births_D_short_w3.png) |
| 31 | `loop_seattle/H/short` | Transport | `H` | 48 | 3266 | 1.39 | 5.91 | **-76.5%** | 0.0218 | 100% | 0.796 | [PNG](univariate_top50/uni_31_loop_seattle_H_short_w3266.png) |
| 32 | `solar/H/short` | Energy | `H` | 48 | 1029 | 5.90 | 13.97 | **-57.7%** | 0.1813 | 92% | 0.918 | [PNG](univariate_top50/uni_32_solar_H_short_w1029.png) |
| 33 | `solar/H/short` | Energy | `H` | 48 | 1853 | 4.87 | 10.57 | **-53.9%** | 0.1444 | 98% | 0.949 | [PNG](univariate_top50/uni_33_solar_H_short_w1853.png) |
| 34 | `solar/H/short` | Energy | `H` | 48 | 1834 | 4.73 | 9.18 | **-48.4%** | 0.1681 | 92% | 0.918 | [PNG](univariate_top50/uni_34_solar_H_short_w1834.png) |
| 35 | `loop_seattle/H/short` | Transport | `H` | 48 | 485 | 3.97 | 11.76 | **-66.3%** | 0.0615 | 96% | 0.815 | [PNG](univariate_top50/uni_35_loop_seattle_H_short_w485.png) |
| 36 | `loop_seattle/5T/long` | Transport | `5T` | 720 | 3728 | 3.15 | 7.26 | **-56.5%** | 0.0478 | 71% | 0.878 | [PNG](univariate_top50/uni_36_loop_seattle_5T_long_w3728.png) |
| 37 | `m4_monthly/M/short` | Econ/Fin | `M` | 18 | 1726 | 28.42 | 159.5 | **-82.2%** | 0.0069 | 100% | 0.728 | [PNG](univariate_top50/uni_37_m4_monthly_M_short_w1726.png) |
| 38 | `loop_seattle/5T/medium` | Transport | `5T` | 480 | 789 | 5.88 | 10.13 | **-41.9%** | 0.0934 | 81% | 0.936 | [PNG](univariate_top50/uni_38_loop_seattle_5T_medium_w789.png) |
| 39 | `loop_seattle/5T/long` | Transport | `5T` | 720 | 3066 | 3.70 | 10.05 | **-63.1%** | 0.0562 | 76% | 0.761 | [PNG](univariate_top50/uni_39_loop_seattle_5T_long_w3066.png) |
| 40 | `solar/10T/medium` | Energy | `10T` | 480 | 1029 | 1.45 | 2.57 | **-43.6%** | 0.2190 | 95% | 0.941 | [PNG](univariate_top50/uni_40_solar_10T_medium_w1029.png) |
| 41 | `m4_weekly/W/short` | Econ/Fin | `W` | 13 | 56 | 132.6 | 573.8 | **-76.9%** | 0.0566 | 92% | 0.620 | [PNG](univariate_top50/uni_41_m4_weekly_W_short_w56.png) |
| 42 | `kdd_cup_2018/H/short` | Nature | `H` | 48 | 3612 | 9.69 | 49.54 | **-80.4%** | 0.2682 | 100% | 0.662 | [PNG](univariate_top50/uni_42_kdd_cup_2018_H_short_w3612.png) |
| 43 | `m4_weekly/W/short` | Econ/Fin | `W` | 13 | 244 | 117.6 | 312.7 | **-62.4%** | 0.0374 | 100% | 0.800 | [PNG](univariate_top50/uni_43_m4_weekly_W_short_w244.png) |
| 44 | `restaurant/D/short` | Sales | `D` | 30 | 272 | 5.77 | 13.19 | **-56.3%** | 0.1301 | 92% | 0.750 | [PNG](univariate_top50/uni_44_restaurant_D_short_w272.png) |
| 45 | `solar/10T/medium` | Energy | `10T` | 480 | 1007 | 1.79 | 2.72 | **-34.3%** | 0.2445 | 96% | 0.935 | [PNG](univariate_top50/uni_45_solar_10T_medium_w1007.png) |
| 46 | `solar/10T/medium` | Energy | `10T` | 480 | 725 | 1.64 | 3.51 | **-53.4%** | 0.8559 | 89% | 0.725 | [PNG](univariate_top50/uni_46_solar_10T_medium_w725.png) |
| 47 | `saugeen/M/short` | Nature | `M` | 12 | 5 | 9.29 | 23.37 | **-60.2%** | 0.2672 | 92% | 0.662 | [PNG](univariate_top50/uni_47_saugeen_M_short_w5.png) |
| 48 | `kdd_cup_2018/H/short` | Nature | `H` | 48 | 776 | 17.81 | 40.65 | **-56.2%** | 0.2042 | 100% | 0.640 | [PNG](univariate_top50/uni_48_kdd_cup_2018_H_short_w776.png) |
| 49 | `kdd_cup_2018/H/short` | Nature | `H` | 48 | 2602 | 10.40 | 20.26 | **-48.6%** | 0.2085 | 100% | 0.663 | [PNG](univariate_top50/uni_49_kdd_cup_2018_H_short_w2602.png) |
| 50 | `restaurant/D/short` | Sales | `D` | 30 | 40 | 7.21 | 12.52 | **-42.4%** | 0.1717 | 86% | 0.668 | [PNG](univariate_top50/uni_50_restaurant_D_short_w40.png) |

<details>
<summary><b>Gallery: View All 50 Univariate Plates Inline</b></summary>

### Univariate #01 — `electricity/D/short` (Window #1835, -92.2% MAE vs. SN, `R^2=0.973`)

![Univariate #01](univariate_top50/uni_01_electricity_D_short_w1835.png)

### Univariate #02 — `m4_yearly/A/short` (Window #9420, -91.1% MAE vs. SN, `R^2=0.956`)

![Univariate #02](univariate_top50/uni_02_m4_yearly_A_short_w9420.png)

### Univariate #03 — `m4_yearly/A/short` (Window #22146, -90.6% MAE vs. SN, `R^2=0.962`)

![Univariate #03](univariate_top50/uni_03_m4_yearly_A_short_w22146.png)

### Univariate #04 — `m4_weekly/W/short` (Window #242, -85.8% MAE vs. SN, `R^2=0.935`)

![Univariate #04](univariate_top50/uni_04_m4_weekly_W_short_w242.png)

### Univariate #05 — `m4_yearly/A/short` (Window #2148, -89.5% MAE vs. SN, `R^2=0.947`)

![Univariate #05](univariate_top50/uni_05_m4_yearly_A_short_w2148.png)

### Univariate #06 — `electricity/H/short` (Window #2661, -85.0% MAE vs. SN, `R^2=0.954`)

![Univariate #06](univariate_top50/uni_06_electricity_H_short_w2661.png)

### Univariate #07 — `m4_yearly/A/short` (Window #7272, -87.3% MAE vs. SN, `R^2=0.916`)

![Univariate #07](univariate_top50/uni_07_m4_yearly_A_short_w7272.png)

### Univariate #08 — `electricity/H/short` (Window #2555, -81.9% MAE vs. SN, `R^2=0.899`)

![Univariate #08](univariate_top50/uni_08_electricity_H_short_w2555.png)

### Univariate #09 — `covid_deaths/D/short` (Window #22, -88.0% MAE vs. SN, `R^2=0.898`)

![Univariate #09](univariate_top50/uni_09_covid_deaths_D_short_w22.png)

### Univariate #10 — `electricity/15T/short` (Window #2661, -83.6% MAE vs. SN, `R^2=0.892`)

![Univariate #10](univariate_top50/uni_10_electricity_15T_short_w2661.png)

### Univariate #11 — `us_births/D/short` (Window #10, -86.0% MAE vs. SN, `R^2=0.895`)

![Univariate #11](univariate_top50/uni_11_us_births_D_short_w10.png)

### Univariate #12 — `covid_deaths/D/short` (Window #194, -87.5% MAE vs. SN, `R^2=0.891`)

![Univariate #12](univariate_top50/uni_12_covid_deaths_D_short_w194.png)

### Univariate #13 — `m4_hourly/H/short` (Window #35, -73.2% MAE vs. SN, `R^2=0.956`)

![Univariate #13](univariate_top50/uni_13_m4_hourly_H_short_w35.png)

### Univariate #14 — `saugeen/W/short` (Window #16, -92.5% MAE vs. SN, `R^2=0.878`)

![Univariate #14](univariate_top50/uni_14_saugeen_W_short_w16.png)

### Univariate #15 — `m4_hourly/H/short` (Window #145, -72.6% MAE vs. SN, `R^2=0.921`)

![Univariate #15](univariate_top50/uni_15_m4_hourly_H_short_w145.png)

### Univariate #16 — `electricity/D/short` (Window #478, -83.3% MAE vs. SN, `R^2=0.848`)

![Univariate #16](univariate_top50/uni_16_electricity_D_short_w478.png)

### Univariate #17 — `electricity/15T/medium` (Window #2661, -73.4% MAE vs. SN, `R^2=0.899`)

![Univariate #17](univariate_top50/uni_17_electricity_15T_medium_w2661.png)

### Univariate #18 — `m4_hourly/H/short` (Window #29, -67.4% MAE vs. SN, `R^2=0.972`)

![Univariate #18](univariate_top50/uni_18_m4_hourly_H_short_w29.png)

### Univariate #19 — `m4_hourly/H/short` (Window #38, -67.0% MAE vs. SN, `R^2=0.971`)

![Univariate #19](univariate_top50/uni_19_m4_hourly_H_short_w38.png)

### Univariate #20 — `loop_seattle/5T/medium` (Window #4972, -78.0% MAE vs. SN, `R^2=0.851`)

![Univariate #20](univariate_top50/uni_20_loop_seattle_5T_medium_w4972.png)

### Univariate #21 — `us_births/D/short` (Window #11, -75.1% MAE vs. SN, `R^2=0.896`)

![Univariate #21](univariate_top50/uni_21_us_births_D_short_w11.png)

### Univariate #22 — `electricity/H/long` (Window #1835, -77.6% MAE vs. SN, `R^2=0.876`)

![Univariate #22](univariate_top50/uni_22_electricity_H_long_w1835.png)

### Univariate #23 — `electricity/15T/medium` (Window #2448, -74.3% MAE vs. SN, `R^2=0.903`)

![Univariate #23](univariate_top50/uni_23_electricity_15T_medium_w2448.png)

### Univariate #24 — `us_births/D/short` (Window #5, -70.5% MAE vs. SN, `R^2=0.910`)

![Univariate #24](univariate_top50/uni_24_us_births_D_short_w5.png)

### Univariate #25 — `m4_quarterly/Q/short` (Window #10013, -85.4% MAE vs. SN, `R^2=0.802`)

![Univariate #25](univariate_top50/uni_25_m4_quarterly_Q_short_w10013.png)

### Univariate #26 — `m4_quarterly/Q/short` (Window #4316, -84.6% MAE vs. SN, `R^2=0.826`)

![Univariate #26](univariate_top50/uni_26_m4_quarterly_Q_short_w4316.png)

### Univariate #27 — `m4_quarterly/Q/short` (Window #8460, -84.7% MAE vs. SN, `R^2=0.741`)

![Univariate #27](univariate_top50/uni_27_m4_quarterly_Q_short_w8460.png)

### Univariate #28 — `covid_deaths/D/short` (Window #135, -84.2% MAE vs. SN, `R^2=0.806`)

![Univariate #28](univariate_top50/uni_28_covid_deaths_D_short_w135.png)

### Univariate #29 — `covid_deaths/D/short` (Window #106, -83.2% MAE vs. SN, `R^2=0.820`)

![Univariate #29](univariate_top50/uni_29_covid_deaths_D_short_w106.png)

### Univariate #30 — `us_births/D/short` (Window #3, -81.3% MAE vs. SN, `R^2=0.795`)

![Univariate #30](univariate_top50/uni_30_us_births_D_short_w3.png)

### Univariate #31 — `loop_seattle/H/short` (Window #3266, -76.5% MAE vs. SN, `R^2=0.796`)

![Univariate #31](univariate_top50/uni_31_loop_seattle_H_short_w3266.png)

### Univariate #32 — `solar/H/short` (Window #1029, -57.7% MAE vs. SN, `R^2=0.918`)

![Univariate #32](univariate_top50/uni_32_solar_H_short_w1029.png)

### Univariate #33 — `solar/H/short` (Window #1853, -53.9% MAE vs. SN, `R^2=0.949`)

![Univariate #33](univariate_top50/uni_33_solar_H_short_w1853.png)

### Univariate #34 — `solar/H/short` (Window #1834, -48.4% MAE vs. SN, `R^2=0.918`)

![Univariate #34](univariate_top50/uni_34_solar_H_short_w1834.png)

### Univariate #35 — `loop_seattle/H/short` (Window #485, -66.3% MAE vs. SN, `R^2=0.815`)

![Univariate #35](univariate_top50/uni_35_loop_seattle_H_short_w485.png)

### Univariate #36 — `loop_seattle/5T/long` (Window #3728, -56.5% MAE vs. SN, `R^2=0.878`)

![Univariate #36](univariate_top50/uni_36_loop_seattle_5T_long_w3728.png)

### Univariate #37 — `m4_monthly/M/short` (Window #1726, -82.2% MAE vs. SN, `R^2=0.728`)

![Univariate #37](univariate_top50/uni_37_m4_monthly_M_short_w1726.png)

### Univariate #38 — `loop_seattle/5T/medium` (Window #789, -41.9% MAE vs. SN, `R^2=0.936`)

![Univariate #38](univariate_top50/uni_38_loop_seattle_5T_medium_w789.png)

### Univariate #39 — `loop_seattle/5T/long` (Window #3066, -63.1% MAE vs. SN, `R^2=0.761`)

![Univariate #39](univariate_top50/uni_39_loop_seattle_5T_long_w3066.png)

### Univariate #40 — `solar/10T/medium` (Window #1029, -43.6% MAE vs. SN, `R^2=0.941`)

![Univariate #40](univariate_top50/uni_40_solar_10T_medium_w1029.png)

### Univariate #41 — `m4_weekly/W/short` (Window #56, -76.9% MAE vs. SN, `R^2=0.620`)

![Univariate #41](univariate_top50/uni_41_m4_weekly_W_short_w56.png)

### Univariate #42 — `kdd_cup_2018/H/short` (Window #3612, -80.4% MAE vs. SN, `R^2=0.662`)

![Univariate #42](univariate_top50/uni_42_kdd_cup_2018_H_short_w3612.png)

### Univariate #43 — `m4_weekly/W/short` (Window #244, -62.4% MAE vs. SN, `R^2=0.800`)

![Univariate #43](univariate_top50/uni_43_m4_weekly_W_short_w244.png)

### Univariate #44 — `restaurant/D/short` (Window #272, -56.3% MAE vs. SN, `R^2=0.750`)

![Univariate #44](univariate_top50/uni_44_restaurant_D_short_w272.png)

### Univariate #45 — `solar/10T/medium` (Window #1007, -34.3% MAE vs. SN, `R^2=0.935`)

![Univariate #45](univariate_top50/uni_45_solar_10T_medium_w1007.png)

### Univariate #46 — `solar/10T/medium` (Window #725, -53.4% MAE vs. SN, `R^2=0.725`)

![Univariate #46](univariate_top50/uni_46_solar_10T_medium_w725.png)

### Univariate #47 — `saugeen/M/short` (Window #5, -60.2% MAE vs. SN, `R^2=0.662`)

![Univariate #47](univariate_top50/uni_47_saugeen_M_short_w5.png)

### Univariate #48 — `kdd_cup_2018/H/short` (Window #776, -56.2% MAE vs. SN, `R^2=0.640`)

![Univariate #48](univariate_top50/uni_48_kdd_cup_2018_H_short_w776.png)

### Univariate #49 — `kdd_cup_2018/H/short` (Window #2602, -48.6% MAE vs. SN, `R^2=0.663`)

![Univariate #49](univariate_top50/uni_49_kdd_cup_2018_H_short_w2602.png)

### Univariate #50 — `restaurant/D/short` (Window #40, -42.4% MAE vs. SN, `R^2=0.668`)

![Univariate #50](univariate_top50/uni_50_restaurant_D_short_w40.png)

</details>

---

## Part II: Top 50 Multivariate & Covariate Ablation Examples

| # | Dataset Config | Family | Channels (`V_tgt+V_cov`) | Win / Var | `MAE` WITHOUT | `MAE` WITH | Δ MAE | `wQL` WITH | `Cov80` | `R^2` | Plot |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | `bizitobs_l2c/H/short` | bizitobs_l2c | `7+2` | `w0/v4` | 1.90 | 1.42 | **-25.1%** | 0.2674 | 100% | 0.883 | [PNG](multivariate_top50/mv_01_bizitobs_l2c_H_short_w0_v4.png) |
| 02 | `bitbrains_rnd/H/short` | bitbrains_rnd | `2+5` | `w353/v1` | 0.051 | 0.032 | **-36.5%** | 0.2486 | 98% | 0.483 | [PNG](multivariate_top50/mv_02_bitbrains_rnd_H_short_w353_v1.png) |
| 03 | `bizitobs_l2c/H/short` | bizitobs_l2c | `7+2` | `w2/v2` | 2.61 | 2.00 | **-23.6%** | 0.2457 | 100% | 0.886 | [PNG](multivariate_top50/mv_03_bizitobs_l2c_H_short_w2_v2.png) |
| 04 | `bitbrains_rnd/5T/long` | bitbrains_rnd | `2+5` | `w268/v1` | 18.51 | 9.93 | **-46.3%** | 0.6970 | 99% | 0.371 | [PNG](multivariate_top50/mv_04_bitbrains_rnd_5T_long_w268_v1.png) |
| 05 | `bizitobs_service/10S/short` | bizitobs_service | `2+34` | `w24/v0` | 11.11 | 9.87 | **-11.1%** | 0.1621 | 100% | 0.701 | [PNG](multivariate_top50/mv_05_bizitobs_service_10S_short_w24_v0.png) |
| 06 | `jena_weather/H/short` | jena_weather | `21+0` | `w7/v0` | 1.51 | 0.991 | **-34.5%** | 0.0013 | 100% | 0.551 | [PNG](multivariate_top50/mv_06_jena_weather_H_short_w7_v0.png) |
| 07 | `jena_weather/10T/long` | jena_weather | `21+0` | `w1/v19` | 2.53 | 2.05 | **-19.3%** | 0.1269 | 98% | 0.837 | [PNG](multivariate_top50/mv_07_jena_weather_10T_long_w1_v19.png) |
| 08 | `jena_weather/10T/medium` | jena_weather | `21+0` | `w0/v0` | 1.03 | 0.812 | **-20.9%** | 0.0012 | 100% | 0.728 | [PNG](multivariate_top50/mv_08_jena_weather_10T_medium_w0_v0.png) |
| 09 | `ett1/15T/short` | ett1 | `7+0` | `w8/v2` | 0.852 | 0.682 | **-20.0%** | 0.0922 | 100% | 0.824 | [PNG](multivariate_top50/mv_09_ett1_15T_short_w8_v2.png) |
| 10 | `bitbrains_rnd/5T/long` | bitbrains_rnd | `2+5` | `w286/v1` | 20.50 | 13.16 | **-35.8%** | 0.8128 | 99% | 0.384 | [PNG](multivariate_top50/mv_10_bitbrains_rnd_5T_long_w286_v1.png) |
| 11 | `ett1/15T/short` | ett1 | `7+0` | `w8/v0` | 0.861 | 0.713 | **-17.3%** | 0.0628 | 100% | 0.839 | [PNG](multivariate_top50/mv_11_ett1_15T_short_w8_v0.png) |
| 12 | `jena_weather/H/short` | jena_weather | `21+0` | `w1/v0` | 0.407 | 0.284 | **-30.3%** | 0.0012 | 100% | 0.352 | [PNG](multivariate_top50/mv_12_jena_weather_H_short_w1_v0.png) |
| 13 | `ett2/H/short` | ett2 | `7+0` | `w14/v6` | 7.43 | 5.85 | **-21.3%** | 0.0343 | 100% | 0.776 | [PNG](multivariate_top50/mv_13_ett2_H_short_w14_v6.png) |
| 14 | `ett1/15T/short` | ett1 | `7+0` | `w12/v4` | 0.449 | 0.379 | **-15.6%** | 0.0812 | 88% | 0.785 | [PNG](multivariate_top50/mv_14_ett1_15T_short_w12_v4.png) |
| 15 | `ett1/15T/short` | ett1 | `7+0` | `w6/v4` | 0.458 | 0.387 | **-15.4%** | 0.0805 | 85% | 0.759 | [PNG](multivariate_top50/mv_15_ett1_15T_short_w6_v4.png) |
| 16 | `bizitobs_service/10S/short` | bizitobs_service | `2+34` | `w171/v0` | 0.289 | 0.205 | **-28.9%** | 0.0156 | 100% | 0.139 | [PNG](multivariate_top50/mv_16_bizitobs_service_10S_short_w171_v0.png) |
| 17 | `bitbrains_fast_storage/5T/short` | bitbrains_fast_storage | `2+5` | `w21626/v0` | 138.9 | 127.9 | **-7.9%** | 0.0502 | 81% | 0.652 | [PNG](multivariate_top50/mv_17_bitbrains_fast_storage_5T_short_w21626_v0.png) |
| 18 | `bitbrains_rnd/H/short` | bitbrains_rnd | `2+5` | `w967/v0` | 1.64 | 1.27 | **-22.4%** | 0.3739 | 96% | 0.228 | [PNG](multivariate_top50/mv_18_bitbrains_rnd_H_short_w967_v0.png) |
| 19 | `bizitobs_service/10S/short` | bizitobs_service | `2+34` | `w170/v0` | 0.506 | 0.409 | **-19.2%** | 0.0176 | 93% | 0.261 | [PNG](multivariate_top50/mv_19_bizitobs_service_10S_short_w170_v0.png) |
| 20 | `jena_weather/10T/short` | jena_weather | `21+0` | `w9/v16` | 25.74 | 23.98 | **-6.8%** | 0.3427 | 85% | 0.716 | [PNG](multivariate_top50/mv_20_jena_weather_10T_short_w9_v16.png) |
| 21 | `jena_weather/10T/long` | jena_weather | `21+0` | `w1/v6` | 0.756 | 0.627 | **-17.0%** | 0.0919 | 100% | 0.630 | [PNG](multivariate_top50/mv_21_jena_weather_10T_long_w1_v6.png) |
| 22 | `jena_weather/10T/short` | jena_weather | `21+0` | `w6/v17` | 32.35 | 22.00 | **-32.0%** | 0.3672 | 100% | 0.353 | [PNG](multivariate_top50/mv_22_jena_weather_10T_short_w6_v17.png) |
| 23 | `jena_weather/10T/short` | jena_weather | `21+0` | `w19/v2` | 0.203 | 0.177 | **-12.9%** | 0.0011 | 100% | 0.690 | [PNG](multivariate_top50/mv_23_jena_weather_10T_short_w19_v2.png) |
| 24 | `bizitobs_l2c/H/short` | bizitobs_l2c | `7+2` | `w4/v5` | 4.89 | 4.39 | **-10.3%** | 0.1391 | 100% | 0.579 | [PNG](multivariate_top50/mv_24_bizitobs_l2c_H_short_w4_v5.png) |
| 25 | `jena_weather/10T/long` | jena_weather | `21+0` | `w2/v17` | 31.67 | 29.11 | **-8.1%** | 0.6067 | 95% | 0.808 | [PNG](multivariate_top50/mv_25_jena_weather_10T_long_w2_v17.png) |
| 26 | `bitbrains_fast_storage/H/short` | bitbrains_fast_storage | `2+5` | `w262/v1` | 0.290 | 0.264 | **-9.0%** | 0.3095 | 100% | 0.500 | [PNG](multivariate_top50/mv_26_bitbrains_fast_storage_H_short_w262_v1.png) |
| 27 | `jena_weather/H/short` | jena_weather | `21+0` | `w1/v5` | 0.772 | 0.663 | **-14.1%** | 0.0861 | 100% | 0.679 | [PNG](multivariate_top50/mv_27_jena_weather_H_short_w1_v5.png) |
| 28 | `bizitobs_service/10S/short` | bizitobs_service | `2+34` | `w294/v1` | 850.0 | 685.7 | **-19.3%** | 0.1353 | 100% | 0.296 | [PNG](multivariate_top50/mv_28_bizitobs_service_10S_short_w294_v1.png) |
| 29 | `bizitobs_service/10S/short` | bizitobs_service | `2+34` | `w150/v1` | 3.03 | 2.48 | **-18.1%** | 0.0259 | 92% | 0.264 | [PNG](multivariate_top50/mv_29_bizitobs_service_10S_short_w150_v1.png) |
| 30 | `jena_weather/10T/medium` | jena_weather | `21+0` | `w2/v17` | 29.95 | 28.01 | **-6.5%** | 0.6006 | 96% | 0.760 | [PNG](multivariate_top50/mv_30_jena_weather_10T_medium_w2_v17.png) |
| 31 | `ett2/15T/medium` | ett2 | `7+0` | `w3/v6` | 3.74 | 3.47 | **-7.1%** | 0.0763 | 93% | 0.719 | [PNG](multivariate_top50/mv_31_ett2_15T_medium_w3_v6.png) |
| 32 | `bitbrains_fast_storage/H/short` | bitbrains_fast_storage | `2+5` | `w1663/v1` | 0.055 | 0.048 | **-12.1%** | 0.2919 | 96% | 0.405 | [PNG](multivariate_top50/mv_32_bitbrains_fast_storage_H_short_w1663_v1.png) |
| 33 | `jena_weather/H/short` | jena_weather | `21+0` | `w0/v5` | 1.01 | 0.833 | **-18.0%** | 0.1096 | 100% | 0.522 | [PNG](multivariate_top50/mv_33_jena_weather_H_short_w0_v5.png) |
| 34 | `m_dense/D/short` | m_dense | `1+7` | `w58/v0` | 19.90 | 18.64 | **-6.3%** | 0.0681 | 93% | 0.716 | [PNG](multivariate_top50/mv_34_m_dense_D_short_w58_v0.png) |
| 35 | `ett2/15T/short` | ett2 | `7+0` | `w10/v6` | 1.30 | 1.08 | **-17.2%** | 0.0239 | 100% | 0.548 | [PNG](multivariate_top50/mv_35_ett2_15T_short_w10_v6.png) |
| 36 | `bitbrains_fast_storage/5T/medium` | bitbrains_fast_storage | `2+5` | `w344/v1` | 4.23 | 3.83 | **-9.3%** | 0.6863 | 97% | 0.455 | [PNG](multivariate_top50/mv_36_bitbrains_fast_storage_5T_medium_w344_v1.png) |
| 37 | `bitbrains_fast_storage/H/short` | bitbrains_fast_storage | `2+5` | `w872/v0` | 4.04 | 3.59 | **-11.2%** | 0.0549 | 92% | 0.880 | [PNG](multivariate_top50/mv_37_bitbrains_fast_storage_H_short_w872_v0.png) |
| 38 | `m_dense/D/short` | m_dense | `1+7` | `w82/v0` | 144.6 | 130.5 | **-9.7%** | 0.0688 | 93% | 0.554 | [PNG](multivariate_top50/mv_38_m_dense_D_short_w82_v0.png) |
| 39 | `bitbrains_rnd/5T/long` | bitbrains_rnd | `2+5` | `w140/v0` | 4798.1 | 3701.5 | **-22.9%** | 1.9246 | 96% | 0.166 | [PNG](multivariate_top50/mv_39_bitbrains_rnd_5T_long_w140_v0.png) |
| 40 | `bitbrains_fast_storage/H/short` | bitbrains_fast_storage | `2+5` | `w447/v0` | 16.44 | 14.66 | **-10.8%** | 0.2365 | 98% | 0.452 | [PNG](multivariate_top50/mv_40_bitbrains_fast_storage_H_short_w447_v0.png) |
| 41 | `bitbrains_fast_storage/5T/short` | bitbrains_fast_storage | `2+5` | `w9883/v1` | 0.043 | 0.036 | **-16.6%** | 0.0191 | 92% | 0.283 | [PNG](multivariate_top50/mv_41_bitbrains_fast_storage_5T_short_w9883_v1.png) |
| 42 | `bitbrains_fast_storage/5T/short` | bitbrains_fast_storage | `2+5` | `w13938/v1` | 0.113 | 0.099 | **-11.9%** | 0.0665 | 100% | 0.165 | [PNG](multivariate_top50/mv_42_bitbrains_fast_storage_5T_short_w13938_v1.png) |
| 43 | `bitbrains_rnd/5T/short` | bitbrains_rnd | `2+5` | `w2117/v0` | 5.06 | 3.98 | **-21.3%** | 0.0254 | 90% | 0.176 | [PNG](multivariate_top50/mv_43_bitbrains_rnd_5T_short_w2117_v0.png) |
| 44 | `bitbrains_rnd/H/short` | bitbrains_rnd | `2+5` | `w53/v1` | 0.053 | 0.042 | **-20.7%** | 0.3448 | 98% | 0.260 | [PNG](multivariate_top50/mv_44_bitbrains_rnd_H_short_w53_v1.png) |
| 45 | `ett2/H/short` | ett2 | `7+0` | `w18/v6` | 11.62 | 10.68 | **-8.1%** | 0.0572 | 90% | 0.372 | [PNG](multivariate_top50/mv_45_ett2_H_short_w18_v6.png) |
| 46 | `ett1/15T/medium` | ett1 | `7+0` | `w0/v1` | 0.895 | 0.838 | **-6.3%** | 0.5281 | 96% | 0.373 | [PNG](multivariate_top50/mv_46_ett1_15T_medium_w0_v1.png) |
| 47 | `bitbrains_rnd/5T/long` | bitbrains_rnd | `2+5` | `w488/v0` | 4370.7 | 3257.1 | **-25.5%** | 0.6565 | 100% | 0.364 | [PNG](multivariate_top50/mv_47_bitbrains_rnd_5T_long_w488_v0.png) |
| 48 | `ett1/15T/long` | ett1 | `7+0` | `w0/v1` | 0.913 | 0.847 | **-7.2%** | 0.4900 | 97% | 0.301 | [PNG](multivariate_top50/mv_48_ett1_15T_long_w0_v1.png) |
| 49 | `bitbrains_rnd/5T/short` | bitbrains_rnd | `2+5` | `w4966/v1` | 0.055 | 0.049 | **-10.0%** | 0.0207 | 96% | 0.388 | [PNG](multivariate_top50/mv_49_bitbrains_rnd_5T_short_w4966_v1.png) |
| 50 | `bitbrains_fast_storage/5T/short` | bitbrains_fast_storage | `2+5` | `w8532/v0` | 6.22 | 5.55 | **-10.7%** | 0.0227 | 100% | 0.174 | [PNG](multivariate_top50/mv_50_bitbrains_fast_storage_5T_short_w8532_v0.png) |

<details>
<summary><b>Gallery: View All 50 Multivariate / Covariate Ablation Plates Inline</b></summary>

### Multivariate #01 — `bizitobs_l2c/H/short` (Window #0, Var #4, `V=7+2`, -25.1% MAE)

![Multivariate #01](multivariate_top50/mv_01_bizitobs_l2c_H_short_w0_v4.png)

### Multivariate #02 — `bitbrains_rnd/H/short` (Window #353, Var #1, `V=2+5`, -36.5% MAE)

![Multivariate #02](multivariate_top50/mv_02_bitbrains_rnd_H_short_w353_v1.png)

### Multivariate #03 — `bizitobs_l2c/H/short` (Window #2, Var #2, `V=7+2`, -23.6% MAE)

![Multivariate #03](multivariate_top50/mv_03_bizitobs_l2c_H_short_w2_v2.png)

### Multivariate #04 — `bitbrains_rnd/5T/long` (Window #268, Var #1, `V=2+5`, -46.3% MAE)

![Multivariate #04](multivariate_top50/mv_04_bitbrains_rnd_5T_long_w268_v1.png)

### Multivariate #05 — `bizitobs_service/10S/short` (Window #24, Var #0, `V=2+34`, -11.1% MAE)

![Multivariate #05](multivariate_top50/mv_05_bizitobs_service_10S_short_w24_v0.png)

### Multivariate #06 — `jena_weather/H/short` (Window #7, Var #0, `V=21+0`, -34.5% MAE)

![Multivariate #06](multivariate_top50/mv_06_jena_weather_H_short_w7_v0.png)

### Multivariate #07 — `jena_weather/10T/long` (Window #1, Var #19, `V=21+0`, -19.3% MAE)

![Multivariate #07](multivariate_top50/mv_07_jena_weather_10T_long_w1_v19.png)

### Multivariate #08 — `jena_weather/10T/medium` (Window #0, Var #0, `V=21+0`, -20.9% MAE)

![Multivariate #08](multivariate_top50/mv_08_jena_weather_10T_medium_w0_v0.png)

### Multivariate #09 — `ett1/15T/short` (Window #8, Var #2, `V=7+0`, -20.0% MAE)

![Multivariate #09](multivariate_top50/mv_09_ett1_15T_short_w8_v2.png)

### Multivariate #10 — `bitbrains_rnd/5T/long` (Window #286, Var #1, `V=2+5`, -35.8% MAE)

![Multivariate #10](multivariate_top50/mv_10_bitbrains_rnd_5T_long_w286_v1.png)

### Multivariate #11 — `ett1/15T/short` (Window #8, Var #0, `V=7+0`, -17.3% MAE)

![Multivariate #11](multivariate_top50/mv_11_ett1_15T_short_w8_v0.png)

### Multivariate #12 — `jena_weather/H/short` (Window #1, Var #0, `V=21+0`, -30.3% MAE)

![Multivariate #12](multivariate_top50/mv_12_jena_weather_H_short_w1_v0.png)

### Multivariate #13 — `ett2/H/short` (Window #14, Var #6, `V=7+0`, -21.3% MAE)

![Multivariate #13](multivariate_top50/mv_13_ett2_H_short_w14_v6.png)

### Multivariate #14 — `ett1/15T/short` (Window #12, Var #4, `V=7+0`, -15.6% MAE)

![Multivariate #14](multivariate_top50/mv_14_ett1_15T_short_w12_v4.png)

### Multivariate #15 — `ett1/15T/short` (Window #6, Var #4, `V=7+0`, -15.4% MAE)

![Multivariate #15](multivariate_top50/mv_15_ett1_15T_short_w6_v4.png)

### Multivariate #16 — `bizitobs_service/10S/short` (Window #171, Var #0, `V=2+34`, -28.9% MAE)

![Multivariate #16](multivariate_top50/mv_16_bizitobs_service_10S_short_w171_v0.png)

### Multivariate #17 — `bitbrains_fast_storage/5T/short` (Window #21626, Var #0, `V=2+5`, -7.9% MAE)

![Multivariate #17](multivariate_top50/mv_17_bitbrains_fast_storage_5T_short_w21626_v0.png)

### Multivariate #18 — `bitbrains_rnd/H/short` (Window #967, Var #0, `V=2+5`, -22.4% MAE)

![Multivariate #18](multivariate_top50/mv_18_bitbrains_rnd_H_short_w967_v0.png)

### Multivariate #19 — `bizitobs_service/10S/short` (Window #170, Var #0, `V=2+34`, -19.2% MAE)

![Multivariate #19](multivariate_top50/mv_19_bizitobs_service_10S_short_w170_v0.png)

### Multivariate #20 — `jena_weather/10T/short` (Window #9, Var #16, `V=21+0`, -6.8% MAE)

![Multivariate #20](multivariate_top50/mv_20_jena_weather_10T_short_w9_v16.png)

### Multivariate #21 — `jena_weather/10T/long` (Window #1, Var #6, `V=21+0`, -17.0% MAE)

![Multivariate #21](multivariate_top50/mv_21_jena_weather_10T_long_w1_v6.png)

### Multivariate #22 — `jena_weather/10T/short` (Window #6, Var #17, `V=21+0`, -32.0% MAE)

![Multivariate #22](multivariate_top50/mv_22_jena_weather_10T_short_w6_v17.png)

### Multivariate #23 — `jena_weather/10T/short` (Window #19, Var #2, `V=21+0`, -12.9% MAE)

![Multivariate #23](multivariate_top50/mv_23_jena_weather_10T_short_w19_v2.png)

### Multivariate #24 — `bizitobs_l2c/H/short` (Window #4, Var #5, `V=7+2`, -10.3% MAE)

![Multivariate #24](multivariate_top50/mv_24_bizitobs_l2c_H_short_w4_v5.png)

### Multivariate #25 — `jena_weather/10T/long` (Window #2, Var #17, `V=21+0`, -8.1% MAE)

![Multivariate #25](multivariate_top50/mv_25_jena_weather_10T_long_w2_v17.png)

### Multivariate #26 — `bitbrains_fast_storage/H/short` (Window #262, Var #1, `V=2+5`, -9.0% MAE)

![Multivariate #26](multivariate_top50/mv_26_bitbrains_fast_storage_H_short_w262_v1.png)

### Multivariate #27 — `jena_weather/H/short` (Window #1, Var #5, `V=21+0`, -14.1% MAE)

![Multivariate #27](multivariate_top50/mv_27_jena_weather_H_short_w1_v5.png)

### Multivariate #28 — `bizitobs_service/10S/short` (Window #294, Var #1, `V=2+34`, -19.3% MAE)

![Multivariate #28](multivariate_top50/mv_28_bizitobs_service_10S_short_w294_v1.png)

### Multivariate #29 — `bizitobs_service/10S/short` (Window #150, Var #1, `V=2+34`, -18.1% MAE)

![Multivariate #29](multivariate_top50/mv_29_bizitobs_service_10S_short_w150_v1.png)

### Multivariate #30 — `jena_weather/10T/medium` (Window #2, Var #17, `V=21+0`, -6.5% MAE)

![Multivariate #30](multivariate_top50/mv_30_jena_weather_10T_medium_w2_v17.png)

### Multivariate #31 — `ett2/15T/medium` (Window #3, Var #6, `V=7+0`, -7.1% MAE)

![Multivariate #31](multivariate_top50/mv_31_ett2_15T_medium_w3_v6.png)

### Multivariate #32 — `bitbrains_fast_storage/H/short` (Window #1663, Var #1, `V=2+5`, -12.1% MAE)

![Multivariate #32](multivariate_top50/mv_32_bitbrains_fast_storage_H_short_w1663_v1.png)

### Multivariate #33 — `jena_weather/H/short` (Window #0, Var #5, `V=21+0`, -18.0% MAE)

![Multivariate #33](multivariate_top50/mv_33_jena_weather_H_short_w0_v5.png)

### Multivariate #34 — `m_dense/D/short` (Window #58, Var #0, `V=1+7`, -6.3% MAE)

![Multivariate #34](multivariate_top50/mv_34_m_dense_D_short_w58_v0.png)

### Multivariate #35 — `ett2/15T/short` (Window #10, Var #6, `V=7+0`, -17.2% MAE)

![Multivariate #35](multivariate_top50/mv_35_ett2_15T_short_w10_v6.png)

### Multivariate #36 — `bitbrains_fast_storage/5T/medium` (Window #344, Var #1, `V=2+5`, -9.3% MAE)

![Multivariate #36](multivariate_top50/mv_36_bitbrains_fast_storage_5T_medium_w344_v1.png)

### Multivariate #37 — `bitbrains_fast_storage/H/short` (Window #872, Var #0, `V=2+5`, -11.2% MAE)

![Multivariate #37](multivariate_top50/mv_37_bitbrains_fast_storage_H_short_w872_v0.png)

### Multivariate #38 — `m_dense/D/short` (Window #82, Var #0, `V=1+7`, -9.7% MAE)

![Multivariate #38](multivariate_top50/mv_38_m_dense_D_short_w82_v0.png)

### Multivariate #39 — `bitbrains_rnd/5T/long` (Window #140, Var #0, `V=2+5`, -22.9% MAE)

![Multivariate #39](multivariate_top50/mv_39_bitbrains_rnd_5T_long_w140_v0.png)

### Multivariate #40 — `bitbrains_fast_storage/H/short` (Window #447, Var #0, `V=2+5`, -10.8% MAE)

![Multivariate #40](multivariate_top50/mv_40_bitbrains_fast_storage_H_short_w447_v0.png)

### Multivariate #41 — `bitbrains_fast_storage/5T/short` (Window #9883, Var #1, `V=2+5`, -16.6% MAE)

![Multivariate #41](multivariate_top50/mv_41_bitbrains_fast_storage_5T_short_w9883_v1.png)

### Multivariate #42 — `bitbrains_fast_storage/5T/short` (Window #13938, Var #1, `V=2+5`, -11.9% MAE)

![Multivariate #42](multivariate_top50/mv_42_bitbrains_fast_storage_5T_short_w13938_v1.png)

### Multivariate #43 — `bitbrains_rnd/5T/short` (Window #2117, Var #0, `V=2+5`, -21.3% MAE)

![Multivariate #43](multivariate_top50/mv_43_bitbrains_rnd_5T_short_w2117_v0.png)

### Multivariate #44 — `bitbrains_rnd/H/short` (Window #53, Var #1, `V=2+5`, -20.7% MAE)

![Multivariate #44](multivariate_top50/mv_44_bitbrains_rnd_H_short_w53_v1.png)

### Multivariate #45 — `ett2/H/short` (Window #18, Var #6, `V=7+0`, -8.1% MAE)

![Multivariate #45](multivariate_top50/mv_45_ett2_H_short_w18_v6.png)

### Multivariate #46 — `ett1/15T/medium` (Window #0, Var #1, `V=7+0`, -6.3% MAE)

![Multivariate #46](multivariate_top50/mv_46_ett1_15T_medium_w0_v1.png)

### Multivariate #47 — `bitbrains_rnd/5T/long` (Window #488, Var #0, `V=2+5`, -25.5% MAE)

![Multivariate #47](multivariate_top50/mv_47_bitbrains_rnd_5T_long_w488_v0.png)

### Multivariate #48 — `ett1/15T/long` (Window #0, Var #1, `V=7+0`, -7.2% MAE)

![Multivariate #48](multivariate_top50/mv_48_ett1_15T_long_w0_v1.png)

### Multivariate #49 — `bitbrains_rnd/5T/short` (Window #4966, Var #1, `V=2+5`, -10.0% MAE)

![Multivariate #49](multivariate_top50/mv_49_bitbrains_rnd_5T_short_w4966_v1.png)

### Multivariate #50 — `bitbrains_fast_storage/5T/short` (Window #8532, Var #0, `V=2+5`, -10.7% MAE)

![Multivariate #50](multivariate_top50/mv_50_bitbrains_fast_storage_5T_short_w8532_v0.png)

</details>

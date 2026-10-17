# Data

## Day-ahead prices (DE-LU, 2025)

`raw/energy_charts_price_DE-LU_2025.json` holds the 2025 day-ahead prices for
the German-Luxembourg bidding zone in EUR/MWh.

- Original source: Bundesnetzagentur | [SMARD.de](https://www.smard.de/), licence
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Downloaded through the Energy-Charts API
  (`https://api.energy-charts.info/price?bzn=DE-LU&start=2025-01-01&end=2025-12-31`),
  which passes the SMARD data through and credits it. The file is the price
  part of the snapshot in my earlier repo
  [german-energy-operations](https://github.com/MUmairSarwar/german-energy-operations)
  (snapshot sha256 `6352802d…9679`, matched before use). Values are not changed.
- Check against SMARD itself (17 Oct 2026): I downloaded the 2025 DE-LU
  quarter-hour series from smard.de and compared it with this file after
  expanding the hourly prices to quarter-hours. Both have 35,040 values, the
  same sum (3,129,833.16), the same minimum (-250.32) and the same maximum
  (583.40 EUR/MWh).
- Until 30 Sep 2025 the market cleared hourly, from 1 Oct 2025 every 15
  minutes. An hourly price is repeated for its four quarter-hours.

## Household load: BDEW H0 profile

The H0 standard load profile for households (with the BDEW dynamisation
function) comes from the open-source package
[demandlib](https://github.com/oemof/demandlib) (MIT licence), which ships the
BDEW/VDEW profile tables. Holidays are the public holidays in Hesse in 2025;
the profile treats them like Sundays. The profile is used in local wall-clock
time.

In the environment, each household's H0 curve is scaled to its yearly
consumption and every quarter-hour is multiplied by lognormal noise with mean 1
(sigma 0.15 by default), so that households and days are not identical.

## Processed file

`processed/inputs_2025.csv` is built by `python scripts/prepare_data.py`.
Columns: UTC time, local time, local date, price (EUR/MWh) and the H0 energy
for a household using 1000 kWh per year (kWh per quarter-hour).

The two days with a clock change (30 Mar and 26 Oct 2025) do not have 96
quarter-hours and are left out, so 363 days are used.

Solar data from DWD was optional in the plan and is not used.

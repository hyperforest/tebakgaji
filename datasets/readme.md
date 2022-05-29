# Contents

- `data.html`: pure raw data from the [data source](https://predictsalary.com). Retrieved using inspect element and save the HTML.
- `raw_data.csv`: table extracted from `data.html`. Columns are fixed (see schema)
- `clean_data.csv`: cleaned with these steps in order
    - Set lower case of feature with string data type
    - Normalize salary to monthly IDR
    - Fix company name
    - Remove duplicates

Note that in `clean_data.csv`, no outlier were removed. Please see notebook for more details on data cleaning.

# Schema

1. `role`: str. Role/title of the job
2. `company`: str. Not all of the companies are disclosed, only Gojek, Shopee, Tiket.com, Traveloka, Tokopedia, Bukalapak companies are disclosed by default. Undisclosed company is set as `Purchase to unlock 👆`
3. `years_of_exp`: int. Years of experience, ranging from 0 to 20 or more
4. `city`: str. City where the salary owner is living
5. `country`: str. Country code such as "ID", "SG", "MY", "DE"
6. `gender`: str. Male, female, or prefer not to tell
7. `currency`: str. Salary currency code such as "IDR" or "USD"
8. `salary`: int. The salary amount based on the currency and the period
9. `mode`: str. Gross or net salary
10. `period`: str. Monthly or annually paid salary
11. `compensation`: str. Non-cash compensation
12. `verified`: str. Whether the salary owner also attach salary slip during the data submission

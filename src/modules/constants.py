COLUMNS = [
    'role',
    'company',
    'years_of_exp',
    'city',
    'country',
    'gender',
    'currency',
    'salary',
    'mode',
    'period',
    'compensation',
    'verified'
]

COMPANIES = [
    'gojek',
    'shopee',
    'tiket.com',
    'traveloka',
    'tokopedia',
    'bukalapak'
]

CITIES = [
    'jakarta',
    'bogor',
    'depok',
    'tangerang',
    'bekasi',
    'bandung',
    'semarang',
    'yogyakarta',
    'solo',
    'surabaya',
    'malang',
    'denpasar'
]

CURRENCY_RATE = {
    'USD': 14525,
    'SGD': 10501,
    'EUR': 15314,
    'ILS': 4346,
    'INR': 189
}

FIX_ROLE = {
    'analysist': 'analyst',
    'analtytics': 'analytics',
    'dats': 'data',
    'enigneer': 'engineer',
    'executivet': 'executive',
    'instructur': 'instructor',
    'managment': 'management',
    'systemenginner': 'system engineer',
    'yunior': 'junior'
}

# contains of key-values pair of correct city and list of mispelled city
# if a city contains any of the value within the mispelled city list,
# this city is then corrected to correct city (the key)
FIX_CITIES = {
    'jakarta': ['jakara', 'jkt', 'jabodetabek', 'jabotabek', 'jakerta',
                'jakarta barat', 'jakarta timur', 'jakarta utara',
                'jakarta selatan', 'jakarta pusat', 'north jakarta',
                'south jakarta', 'west jakarta', 'east jakarta',
                'central jakarta', 'bogor', 'depok', 'tangerang',
                'bekasi'],
    'yogyakarta': ['yogya', 'jogja', 'jogjakarta', 'yk'],
    'denpasar': ['bali'],
}

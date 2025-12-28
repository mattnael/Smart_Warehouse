import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Konfigurasi Data (Setahun)
num_days = 365
start_date = datetime(2023, 1, 1)

# Generate Data
dates = [start_date + timedelta(days=i) for i in range(num_days)]
# Pola demand acak tapi realistis
demand = [int(50 + 20 * np.sin(i/30) + np.random.normal(0, 5)) for i in range(num_days)]
# waktu tunggu barang (1-7 hari)
lead_time = np.random.randint(1, 8, num_days)
# Tingkat kepentingan barang (1-10)
criticality = np.random.randint(1, 11, num_days)

df = pd.DataFrame({
    'date': dates,
    'demand': demand,
    'lead_time': lead_time,
    'criticality': criticality
})

# Simpan ke Excel
df.to_excel('warehouse_data.xlsx', index=False)
print("File 'warehouse_data.xlsx' berhasil dibuat.")
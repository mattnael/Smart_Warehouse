import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==========================================
# 1. PERSIAPAN DATA
# ==========================================
print("--- Memuat Data ---")
# Membaca file Excel
try:
    df = pd.read_excel('warehouse_data.xlsx')
except FileNotFoundError:
    print("ERROR: File 'warehouse_data.xlsx' tidak ditemukan. Jalankan DummyData.py dulu!")
    exit()

data_demand = df['demand'].values.reshape(-1, 1)

# Normalisasi data menjadi angka 0 sampai 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_demand)

# Membuat urutan data (Sequence 14 hari)
SEQUENCE_LENGTH = 14
X, y = [], []
for i in range(len(scaled_data) - SEQUENCE_LENGTH):
    X.append(scaled_data[i:i+SEQUENCE_LENGTH])
    y.append(scaled_data[i+SEQUENCE_LENGTH])

X, y = np.array(X), np.array(y)

# Bagi data: 80% untuk latihan (Train), 20% untuk ujian (Test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ==========================================
# 2. MEMBUAT & MELATIH MODEL LSTM
# ==========================================
print("\n--- Melatih Model LSTM (Tunggu sebentar...) ---")

model = Sequential()
# Menambahkan layer LSTM
model.add(LSTM(64, return_sequences=False, input_shape=(SEQUENCE_LENGTH, 1)))
model.add(Dropout(0.2)) # Mencegah AI terlalu menghafal (Overfitting)
model.add(Dense(1))     # Output layer

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Mulai proses belajar (Training)
# Epochs 50 = Membaca data 50 kali ulang
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluasi kepintaran AI
predictions = model.predict(X_test)
predictions_inv = scaler.inverse_transform(predictions) # Kembalikan ke angka asli
y_test_inv = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_inv, predictions_inv)
print(f"\nRata-rata kesalahan prediksi (MAE): {mae:.2f} unit barang")

# ==========================================
# 3. SISTEM FUZZY LOGIC (PENGAMBIL KEPUTUSAN)
# ==========================================
print("\n--- Menyiapkan Logika Fuzzy ---")

# Variabel Input
pred_demand = ctrl.Antecedent(np.arange(0, 101, 1), 'predicted_demand')
lead_time = ctrl.Antecedent(np.arange(0, 11, 1), 'lead_time')
criticality = ctrl.Antecedent(np.arange(0, 11, 1), 'criticality')

# Variabel Output
priority = ctrl.Consequent(np.arange(0, 101, 1), 'reorder_priority')

# Membuat kategori otomatis (Rendah, Sedang, Tinggi)
pred_demand.automf(3, names=['Low', 'Medium', 'High'])
lead_time.automf(3, names=['Short', 'Medium', 'Long'])
criticality.automf(3, names=['Low', 'Medium', 'High'])
priority.automf(3, names=['Low', 'Medium', 'High'])

# Aturan Kebijakan Gudang (Rules)
rule1 = ctrl.Rule(pred_demand['High'] & criticality['High'], priority['High'])
rule2 = ctrl.Rule(pred_demand['Medium'] & lead_time['Long'], priority['High'])
rule3 = ctrl.Rule(pred_demand['Low'] & criticality['Low'], priority['Low'])
rule4 = ctrl.Rule(criticality['Medium'], priority['Medium'])
rule5 = ctrl.Rule(pred_demand['High'] & lead_time['Short'], priority['Medium'])

# Membuat sistem kontrol
priority_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
decision_system = ctrl.ControlSystemSimulation(priority_ctrl)

# ==========================================
# 4. SIMULASI AKHIR (INTEGRASI)
# ==========================================
print("\n" + "="*40)
print("   HASIL KEPUTUSAN SMART WAREHOUSE")
print("="*40)

# Ambil data 14 hari terakhir untuk prediksi besok
last_14_days = scaled_data[-SEQUENCE_LENGTH:]
last_14_days = last_14_days.reshape(1, SEQUENCE_LENGTH, 1)

# 1. AI Memprediksi Permintaan Besok
lstm_pred_scaled = model.predict(last_14_days)
final_predicted_demand = scaler.inverse_transform(lstm_pred_scaled)[0][0]

# Data simulasi kondisi gudang saat ini
current_lead_time = 7   # Misal: butuh 7 hari barang sampai
current_criticality = 8 # Misal: barang ini sangat penting (skala 8/10)

# 2. Masukkan ke Fuzzy Logic
decision_system.input['predicted_demand'] = final_predicted_demand
decision_system.input['lead_time'] = current_lead_time
decision_system.input['criticality'] = current_criticality

# Hitung Skor Prioritas
decision_system.compute()
fuzzy_score = decision_system.output['reorder_priority']

# 3. Tentukan Status
status = "AMAN"
if fuzzy_score >= 60:
    status = "⚠️ PERLU RESTOCK SEGERA"
elif fuzzy_score >= 40:
    status = "⚠️ PERTIMBANGKAN RESTOCK"

print(f"Prediksi Permintaan (Esok) : {final_predicted_demand:.2f} unit")
print(f"Waktu Tunggu (Lead Time)   : {current_lead_time} hari")
print(f"Tingkat Kepentingan        : {current_criticality} / 10")
print("-" * 40)
print(f"Skor Prioritas Sistem      : {fuzzy_score:.2f} / 100")
print(f"REKOMENDASI                : {status}")
print("="*40)
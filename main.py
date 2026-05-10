from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
import warnings

# Matikan warning agar terminal bersih
warnings.filterwarnings('ignore')
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

app = FastAPI(
    title="API Prediksi Posyandu (On-The-Fly)",
    description="API untuk melatih AI dan memprediksi pertumbuhan anak secara langsung"
)

# =============================================================
# 1. FUNGSI STATUS GIZI (BAHASA RAMAH IBU-IBU)
# =============================================================
def klasifikasi_status_gizi(z_score, indikator):
    if indikator == 'BB/U':
        if z_score < -3: return "Berat Badan Sangat Kurang (Perlu Konsultasi Bidan/Dokter)"
        elif -3 <= z_score < -2: return "Beresiko Berat Badan Kurang (Ayo Tingkatkan Nutrisi!)"
        elif -2 <= z_score <= 1: return "Berat Badan Normal (Pertumbuhan Aman)"
        else: return "Beresiko Berat Badan Lebih (Waspada Kegemukan)"
        
    elif indikator == 'TB/U':
        if z_score < -3: return "Sangat Pendek (Perlu Perhatian Khusus)"
        elif -3 <= z_score < -2: return "Beresiko Pendek (Waspada Indikasi Stunting)"
        elif -2 <= z_score <= 3: return "Tinggi Badan Normal (Pertumbuhan Aman)"
        else: return "Cenderung Lebih Tinggi dari Umurnya"
        
    elif indikator == 'LK/U':
        if z_score < -2: return "Lingkar Kepala Lebih Kecil dari Normal"
        elif -2 <= z_score <= 2: return "Lingkar Kepala Normal (Aman)"
        else: return "Lingkar Kepala Lebih Besar dari Normal"
        
    return "Tidak Diketahui"

# =============================================================
# 2. SCHEMA INPUT (Menyesuaikan Format Flutter)
# =============================================================
class DataKontrol(BaseModel):
    tanggal: str
    berat: float
    tinggi: float
    lingkar_kepala: float

class RequestPrediksi(BaseModel):
    tanggal_lahir: str
    gender: str
    data_kontrol: List[DataKontrol]
    jumlah_prediksi: int = 3 # Default tebak 3 bulan ke depan sesuai permintaan

    # Contoh data di halaman /docs agar Dosen gampang nge-test
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                  "tanggal_lahir": "2020-01-15",
                  "gender": "L",
                  "data_kontrol": [
                    {"tanggal": "2023-01-15", "berat": 12.0, "tinggi": 90.5, "lingkar_kepala": 48.0},
                    {"tanggal": "2023-02-15", "berat": 12.2, "tinggi": 91.0, "lingkar_kepala": 48.2},
                    {"tanggal": "2023-03-15", "berat": 12.5, "tinggi": 92.1, "lingkar_kepala": 48.5}
                  ],
                  "jumlah_prediksi": 3
                }
            ]
        }
    }

# =============================================================
# 3. ENDPOINT UTAMA (TRAIN & PREDICT ON-THE-FLY)
# =============================================================
@app.post("/prediksi", tags=["Fitur Prediksi AI"])
def lakukan_prediksi(payload: RequestPrediksi):
    # 1. Konversi data JSON dari Flutter ke DataFrame Pandas
    df = pd.DataFrame([vars(d) for d in payload.data_kontrol])
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df = df.sort_values('tanggal')

    # Minimal butuh 2 riwayat untuk bisa bikin garis tren
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Minimal harus ada 2 data kontrol untuk bisa memprediksi.")

    tgl_lahir = pd.to_datetime(payload.tanggal_lahir)
    
    # Tempat menyimpan hasil akhir
    prediksi_dict = {'berat': [], 'tinggi': [], 'lingkar_kepala': []}
    status_dict = {'berat': [], 'tinggi': [], 'lingkar_kepala': []}
    tanggal_list = []

    # 2. LOOPING TRAINING UNTUK KETIGA METRIK SEKALIGUS
    for metrik in ['berat', 'tinggi', 'lingkar_kepala']:
        df_train = df[['tanggal', metrik]].rename(columns={'tanggal': 'ds', metrik: 'y'}).dropna()
        df_train = df_train[df_train['y'] > 0]

        if len(df_train) < 2: continue

        # Set parameter sesuai dengan "Jalan Tengah" yang sudah kita buat
        if metrik == 'berat':
            tipe_growth = 'logistic'
            df_train['cap'] = 35
            kode_ind = 'BB/U'
        elif metrik == 'tinggi':
            tipe_growth = 'linear'
            df_train['cap'] = 130
            kode_ind = 'TB/U'
        else:
            tipe_growth = 'linear'
            df_train['cap'] = 60
            kode_ind = 'LK/U'

        # LANGSUNG TRAINING AI DETIK INI JUGA!
        model = Prophet(
            growth=tipe_growth,
            interval_width=0.95,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.015,
            changepoint_range=0.8
        )
        model.fit(df_train)

        # BUAT TEBAKAN KE DEPAN
        future = model.make_future_dataframe(periods=payload.jumlah_prediksi, freq='MS')
        if tipe_growth == 'logistic':
            future['cap'] = df_train['cap'].iloc[0]

        forecast = model.predict(future)
        
        # Ambil hanya bulan-bulan masa depan (sesuai jumlah_prediksi)
        future_forecast = forecast.iloc[-payload.jumlah_prediksi:].copy()

        # PENGAMAN BIOLOGIS (Tinggi & LK tidak boleh turun/menyusut dari data terakhir!)
        if metrik in ['tinggi', 'lingkar_kepala']:
            last_actual = df_train['y'].iloc[-1]
            preds = np.concatenate(([last_actual], future_forecast['yhat'].values))
            preds = np.maximum.accumulate(preds)[1:] # Paksa grafik tidak bisa turun
            future_forecast['yhat'] = preds

        # Simpan angka tebakan
        prediksi_dict[metrik] = future_forecast['yhat'].values.round(2).tolist()
        
        if len(tanggal_list) == 0:
            tanggal_list = future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist()

        # MENGHITUNG Z-SCORE & STATUS GIZI
        mean_aktual = df_train['y'].mean()
        std_aktual = df_train['y'].std()

        status_list = []
        for val in prediksi_dict[metrik]:
            if std_aktual == 0: z = 0
            else: z = round((val - mean_aktual) / std_aktual, 2)
            z = max(-4, min(4, z))
            status_list.append(klasifikasi_status_gizi(z, kode_ind))
        
        status_dict[metrik] = status_list

    # 3. MERAKIT JAWABAN UNTUK DIKIRIM KEMBALI KE FLUTTER
    hasil_final = []
    for i in range(len(tanggal_list)):
        tgl_pred = pd.to_datetime(tanggal_list[i])
        umur_hari = (tgl_pred - tgl_lahir).days
        umur_bulan = int(umur_hari / 30.44) # Hitung umur anak saat bulan prediksi tersebut!

        data_bulan_ini = {
            "bulan_ke": i + 1,
            "tanggal_prediksi": tanggal_list[i],
            "umur_bulan": umur_bulan,
            "prediksi": {}
        }

        for metrik in ['berat', 'tinggi', 'lingkar_kepala']:
            if len(prediksi_dict[metrik]) > i:
                data_bulan_ini["prediksi"][metrik] = {
                    "nilai_angka": prediksi_dict[metrik][i],
                    "status_gizi": status_dict[metrik][i]
                }
        hasil_final.append(data_bulan_ini)

    return {
        "status": "sukses",
        "pesan": "AI berhasil dilatih dan memprediksi secara real-time",
        "gender_anak": payload.gender,
        "jumlah_prediksi": payload.jumlah_prediksi,
        "hasil_masa_depan": hasil_final
    }

@app.get("/")
def cek_server():
    return {"status": "Server API Skripsi Posyandu Nyala (Versi On-The-Fly)!"}
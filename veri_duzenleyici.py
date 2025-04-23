import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Veri kaynağı: https://www.kaggle.com/datasets/l3llff/step-count-from-phone-app
# Açıklama: Telefon uygulamasından alınan günlük adım sayısı verileri.

def load_data(user_id=4):
    """CSV dosyasını oku ve kullanıcı verilerini filtrele."""
    current_directory = os.getcwd()
    file_path = f'{current_directory}/step-count-from-phone-app.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError("Veri dosyası bulunamadı!")
    df = pd.read_csv(file_path)
    df = df[df["user_id"] == user_id]
    return df

def durum_belirle(value):
    """Adım sayısına göre aktivite durumunu belirle."""
    if value <= 4000:
        return "Düşük Aktivite"
    elif value <= 9000:
        return "Orta Aktivite"
    else:
        return "Yüksek Aktivite"

def turn_to_datetime(df):
    """Tarih kolonlarını datetime formatına çevir."""
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    return df

def gunluk_veriler(df):
    """Günlük bazda verileri gruplar ve durumları belirler."""
    df['gun'] = df['start'].dt.date
    gunluk_df = df.groupby('gun').agg({
        'value': 'sum',
        'start': 'first',
        'end': 'last'
    }).reset_index()
    gunluk_df["durum"] = gunluk_df["value"].apply(durum_belirle)
    gunluk_df = gunluk_df[['gun', 'value', 'durum']]
    return gunluk_df

def filtre_ardisik_gunler(df):
    """Ardışık olmayan günleri veriden çıkar."""
    df = df.sort_values(by='gun')
    df['gun'] = pd.to_datetime(df['gun'])
    df['gun_farki'] = (df['gun'] - df['gun'].shift()).dt.days
    df = df[df['gun_farki'] == 1].drop(columns=['gun_farki'])
    return df

def main():
    """Ana fonksiyon."""
    df = load_data()
    df = turn_to_datetime(df)
    gunluk_df = gunluk_veriler(df)
    gunluk_df = filtre_ardisik_gunler(gunluk_df)
    gunluk_df.to_csv('gunluk_veriler.csv', index=False, encoding='utf-8')
    print("Günlük veriler 'gunluk_veriler.csv' dosyasına kaydedildi.")
    print(f"Bir günde atılan maksimum adım sayısı: {gunluk_df['value'].max()}, Bir günde atılan minimum adım sayısı: {gunluk_df['value'].min()}")
    print(gunluk_df["durum"].value_counts())

if __name__ == "__main__":
    main()
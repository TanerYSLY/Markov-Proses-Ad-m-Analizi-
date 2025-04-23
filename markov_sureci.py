import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """
    Veriyi yükler ve gerekli ön işlemleri yapar.
    """
    current_directory = os.getcwd()
    df = pd.read_csv(f'{current_directory}/gunluk_veriler.csv')
    States = ["Düşük Aktivite", "Orta Aktivite", "Yüksek Aktivite"]

    # Geçiş çiftlerini oluştur
    transitions = list(zip(df['durum'][:-1], df['durum'][1:]))

    # Geçişleri say
    transition_counts = pd.DataFrame(0, index=States, columns=States)
    for (prev, curr) in transitions:
        transition_counts.loc[prev, curr] += 1

    return df, States, transition_counts


def plot_heatmap(df, title="Heatmap", cmap="Blues", annot=True, fmt=".2f"):
    """
    Verilen bir DataFrame için heatmap çizer.

    Args:
        df (pd.DataFrame): Heatmap çizilecek DataFrame.
        title (str): Heatmap başlığı.
        cmap (str): Renk haritası (örn. "Blues", "coolwarm").
        annot (bool): Hücrelere değerleri yazdırmak için True.
        fmt (str): Hücre değerlerinin formatı (örn. ".2f").
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap, xticklabels=df.columns, yticklabels=df.index)
    plt.title(title)
    plt.xlabel("Sütunlar")
    plt.ylabel("Satırlar")
    plt.tight_layout()
    plt.show()


def calculate_stationary_distribution(transition_matrix):
    """
    Geçiş matrisinin denge dağılımını hesaplar.
    """
    eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stationary = np.real(eigvecs[:, idx])
    stationary /= stationary.sum()
    return pd.DataFrame(stationary, index=transition_matrix.index, columns=["Stationary Probability"])


def verify_stationary_distribution(transition_matrix, stationary_distribution):
    """
    Denge dağılımını doğrular: π * P = π denklemini kontrol eder.
    """
    stationary_vector = stationary_distribution.values.flatten()
    return np.allclose(stationary_vector @ transition_matrix.values, stationary_vector)


def soru2(df, States):
    """
    İlk 30 gün için durum grafiğini çizer.
    """
    df['durum_kod'] = df['durum'].astype('category').cat.codes
    df_first_30 = df.head(30)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df_first_30['gun'], y=df_first_30['durum_kod'], drawstyle="steps-post", linewidth=2)
    plt.yticks(ticks=range(len(States)), labels=States)
    plt.xticks(rotation=45)
    plt.xlabel("Günler")
    plt.ylabel("Durum")
    plt.title("İlk 30 Gün İçin Günlere Göre Durum Grafiği")
    plt.grid()
    plt.tight_layout()
    plt.show()


def soru3(transition_counts):
    """
    Geçiş olasılıkları matrisini ve denge dağılımını hesaplar.
    """
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

    print("Geçiş Sayıları Matrisi:\n", transition_counts)
    print("\nGeçiş Olasılıkları Matrisi:\n", transition_matrix)

    plot_heatmap(transition_matrix, title="Geçiş Olasılıkları Heatmap", cmap="Blues", annot=True, fmt=".2f")

    stationary_df = calculate_stationary_distribution(transition_matrix)
    print("\nDenge Dağılımı:\n", stationary_df)

    valid_stationary = verify_stationary_distribution(transition_matrix, stationary_df)
    print("\nDenge dağılımı geçerli mi?:", valid_stationary)


def soru4(transition_counts):
    """
    3, 10 ve 100 adımlı geçiş matrislerini hesaplar ve görselleştirir.
    """
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

    for steps in [3, 10, 100]:
        P = np.linalg.matrix_power(transition_matrix.values, steps)
        P_df = pd.DataFrame(P, index=transition_matrix.index, columns=transition_matrix.columns)
        print(f"\n{steps} günlük Geçiş Matrisi (P^{steps}):\n", P_df)
        plot_heatmap(P_df, title=f"{steps} günlük Geçiş Matrisi (P^{steps})", cmap="Blues", annot=True, fmt=".2f")


def soru6(verify_stationary_distribution, calculate_stationary_distribution, transition_matrix, stationary_distribution):
    """
    Denge dağılımını doğrular ve hesaplar.
    """
    print("Denge dağılımı var mı?", verify_stationary_distribution(transition_matrix, stationary_distribution))
    print("Denge dağılımı:", calculate_stationary_distribution(transition_matrix))


def main():
    """
    Ana fonksiyon: Veriyi yükler ve tüm soruları çalıştırır.
    """
    # Veriyi yükle
    df, States, transition_counts = load_data()

    # Soru 2: İlk 30 gün için durum grafiği
    soru2(df, States)

    # Soru 3: Geçiş olasılıkları matrisi ve denge dağılımı
    soru3(transition_counts)

    # Soru 4: 3, 10 ve 100 adımlı geçiş matrisleri
    soru4(transition_counts)

    # Soru 6: Denge dağılımını doğrula
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
    stationary_distribution = calculate_stationary_distribution(transition_matrix)
    soru6(verify_stationary_distribution, calculate_stationary_distribution, transition_matrix, stationary_distribution)


if __name__ == "__main__":
    main()


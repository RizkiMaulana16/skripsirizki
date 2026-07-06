# ==============================================================================
# APLIKASI STREAMLIT: EVALUASI, CLUSTERING & PEMETAAN SPASIAL RTH
# Diterjemahkan dari script RTH.R (K-Means Integrasi)
# PROGRAM STUDI TEKNIK INFORMATIKA - UNIVERSITAS MUHAMMADIYAH SUKABUMI
# AUTHOR ASLI: MOH RIZKI MAULANA
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import folium
from streamlit_folium import st_folium
from io import StringIO

st.set_page_config(
    page_title="Klasterisasi Zona RTH Kabupaten Sukabumi",
    page_icon="🌳",
    layout="wide",
)

# ------------------------------------------------------------------------------
# BAGIAN 1: SIDEBAR - INPUT DATA
# ------------------------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader(
    "Unggah dataset CSV (opsional)", type=["csv"]
)
st.sidebar.caption(
    "Jika tidak diunggah, aplikasi akan memakai dataset bawaan: `Dataset_rth.csv`"
)

k_optimal = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=8, value=3)

st.title("🌳 Evaluasi, Klasterisasi & Pemetaan Spasial Ruang Terbuka Hijau (RTH)")
st.markdown(
    "Kabupaten Sukabumi — Metode **K-Means Clustering** "
    "(hasil terjemahan dari script R `RTH.R`)"
)

# ------------------------------------------------------------------------------
# BAGIAN 2: PERSIAPAN DATA (DATA PREPARATION)
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_clean(file_or_path):
    df_raw = pd.read_csv(file_or_path)

    # Normalisasi nama kolom (setara clean_names() di R -> snake_case)
    df = df_raw.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Data cleaning: mengatasi anomali tanda baca lokal Indonesia pada teks angka
    # Menghapus koma pemisah ribuan pada luas kecamatan
    df["luas_kec_raw"] = (
        df["luas_kecamatan"].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    # Mengubah koma menjadi titik desimal pada luas ruang terbuka
    df["luas_rth_raw"] = (
        df["luas_ruang_terbuka"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )

    # Konversi satuan Hektar (Ha) -> Km2, serta koreksi skala khusus Kecamatan Cibitung
    df["luas_kec_km2"] = np.where(
        df["nama_kecamatan"].str.lower() == "cibitung",
        88.84,
        df["luas_kec_raw"] / 100,
    )
    df["luas_rth_km2"] = df["luas_rth_raw"]

    # Rekayasa fitur: persentase ketersediaan RTH
    df["persentase_rth"] = (df["luas_rth_km2"] / df["luas_kec_km2"]) * 100

    return df


try:
    if uploaded_file is not None:
        df_clean = load_and_clean(uploaded_file)
        sumber_data = "File yang diunggah"
    else:
        df_clean = load_and_clean("Dataset_rth.csv")
        sumber_data = "Dataset bawaan (Dataset_rth.csv)"
except Exception as e:
    st.error(f"Gagal membaca/membersihkan data: {e}")
    st.stop()

st.success(f"Data berhasil dimuat dari: **{sumber_data}** ({len(df_clean)} kecamatan)")

with st.expander("🔍 Lihat Data Bersih (setelah data preparation)"):
    st.dataframe(
        df_clean[
            ["nama_kecamatan", "latitude", "longitude", "luas_kec_km2", "luas_rth_km2", "persentase_rth"]
        ].round(3),
        use_container_width=True,
    )

# Variabel untuk pemodelan
data_model = df_clean[["luas_kec_km2", "persentase_rth"]].copy()

# Normalisasi Min-Max (0-1)
def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

data_scaled = data_model.apply(min_max_normalize)
data_scaled.index = df_clean["nama_kecamatan"]

# ------------------------------------------------------------------------------
# BAGIAN 3: TAHAP EVALUASI PENENTUAN JUMLAH KLASTER (K)
# ------------------------------------------------------------------------------
st.header("📊 Tahap Evaluasi Penentuan Jumlah Klaster (K)")

k_range = list(range(2, 9))
wcss, silhouette_scores, dbi_scores = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=25)
    labels = km.fit_predict(data_scaled)
    wcss.append(km.inertia_)
    silhouette_scores.append(silhouette_score(data_scaled, labels))
    dbi_scores.append(davies_bouldin_score(data_scaled, labels))

col1, col2, col3 = st.columns(3)

with col1:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=k_range, y=wcss, mode="lines+markers"))
    fig_elbow.update_layout(
        title="Metode Elbow (WCSS)",
        xaxis_title="Jumlah Klaster (K)",
        yaxis_title="Total Within-Cluster Sum of Squares",
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(x=k_range, y=silhouette_scores, mode="lines+markers", line=dict(color="green")))
    fig_sil.update_layout(
        title="Silhouette Analysis",
        xaxis_title="Jumlah Klaster (K)",
        yaxis_title="Rata-rata Lebar Silhouette",
    )
    st.plotly_chart(fig_sil, use_container_width=True)

with col3:
    fig_dbi = go.Figure()
    fig_dbi.add_trace(go.Scatter(x=k_range, y=dbi_scores, mode="lines+markers", line=dict(color="darkred")))
    fig_dbi.update_layout(
        title="Davies-Bouldin Index (DBI)",
        xaxis_title="Jumlah Klaster (K)",
        yaxis_title="Skor DBI",
    )
    st.plotly_chart(fig_dbi, use_container_width=True)

idx_k3 = k_range.index(3)
st.info(f"Nilai Skor Eksak DBI untuk konfigurasi K=3 adalah: **{dbi_scores[idx_k3]:.4f}**")

# ------------------------------------------------------------------------------
# BAGIAN 4: PEMODELAN DETERMINISTIK K-MEANS
# ------------------------------------------------------------------------------
st.header(f"🎯 Pemodelan K-Means (K={k_optimal})")

model_kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=25)
df_clean = df_clean.copy()
df_clean["Cluster_ID"] = model_kmeans.fit_predict(data_scaled) + 1  # +1 agar mulai dari 1 (seperti R)

# ------------------------------------------------------------------------------
# BAGIAN 5: INTERPRETASI DAN LABELING
# ------------------------------------------------------------------------------
centroid_asli = (
    df_clean.groupby("Cluster_ID")[["luas_kec_km2", "persentase_rth"]].mean().reset_index()
)
st.subheader("Nilai Centroid Asli per Klaster")
st.dataframe(centroid_asli.round(3), use_container_width=True)

if k_optimal == 3:
    id_tinggi = centroid_asli.loc[centroid_asli["persentase_rth"].idxmax(), "Cluster_ID"]
    id_rendah = centroid_asli.loc[centroid_asli["persentase_rth"].idxmin(), "Cluster_ID"]
    id_sedang = [c for c in centroid_asli["Cluster_ID"] if c not in (id_tinggi, id_rendah)][0]

    def label_zona(cid):
        if cid == id_tinggi:
            return "Zona RTH Tinggi (Sangat Baik)"
        elif cid == id_sedang:
            return "Zona RTH Sedang (Cukup/Ideal)"
        else:
            return "Zona RTH Rendah (Kritis)"

    df_final = df_clean.copy()
    df_final["Status_Zona"] = df_final["Cluster_ID"].apply(label_zona)
else:
    # Jika K != 3, urutkan label berdasarkan rank persentase_rth
    order = centroid_asli.sort_values("persentase_rth")["Cluster_ID"].tolist()
    label_map = {cid: f"Klaster {i+1} (persentase RTH rank {i+1}/{len(order)})" for i, cid in enumerate(order)}
    df_final = df_clean.copy()
    df_final["Status_Zona"] = df_final["Cluster_ID"].map(label_map)

# ------------------------------------------------------------------------------
# BAGIAN 6: VISUALISASI OUTPUT AKHIR
# ------------------------------------------------------------------------------
st.header("🗺️ Visualisasi Hasil Akhir")

tab1, tab2 = st.tabs(["Scatter Plot Klaster", "Peta Spasial Interaktif"])

with tab1:
    plot_df = data_scaled.copy()
    plot_df["nama_kecamatan"] = df_final["nama_kecamatan"].values
    plot_df["Status_Zona"] = df_final["Status_Zona"].values

    fig_scatter = px.scatter(
        plot_df,
        x="luas_kec_km2",
        y="persentase_rth",
        color="Status_Zona",
        text="nama_kecamatan",
        title=f"Hasil Klasterisasi Zona RTH Kabupaten Sukabumi (K={k_optimal})",
        labels={
            "luas_kec_km2": "Luas Kecamatan (Normalisasi Min-Max)",
            "persentase_rth": "Persentase RTH (Normalisasi Min-Max)",
        },
    )
    fig_scatter.update_traces(textposition="top center", marker=dict(size=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    warna_map = {
        "Zona RTH Rendah (Kritis)": "red",
        "Zona RTH Sedang (Cukup/Ideal)": "orange",
        "Zona RTH Tinggi (Sangat Baik)": "green",
    }

    center_lat = df_final["latitude"].mean()
    center_lon = df_final["longitude"].mean()
    peta_rth = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

    for _, row in df_final.iterrows():
        warna = warna_map.get(row["Status_Zona"], "blue")
        popup_html = f"""
        <div style='font-family: Arial, sans-serif; font-size: 12px;'>
            <strong>KECAMATAN: </strong>{str(row['nama_kecamatan']).upper()}<br>
            <hr style='margin: 4px 0;'>
            <strong>Status Wilayah: </strong><span style='color:blue;'>{row['Status_Zona']}</span><br>
            <strong>Luas Wilayah: </strong>{round(row['luas_kec_km2'], 2)} km²<br>
            <strong>Rasio RTH: </strong>{round(row['persentase_rth'], 2)} %<br>
            <strong>Koordinat: </strong>{row['latitude']}, {row['longitude']}
        </div>
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=9,
            color="black",
            weight=1,
            fill=True,
            fill_color=warna,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row["nama_kecamatan"],
        ).add_to(peta_rth)

    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 9999;
                background-color: white; padding: 10px; border: 1px solid grey; border-radius: 5px;">
        <b>Legenda Zona RTH</b><br>
        <i style="background:green; width:10px; height:10px; display:inline-block;"></i> Zona RTH Tinggi<br>
        <i style="background:orange; width:10px; height:10px; display:inline-block;"></i> Zona RTH Sedang<br>
        <i style="background:red; width:10px; height:10px; display:inline-block;"></i> Zona RTH Rendah
    </div>
    """
    peta_rth.get_root().html.add_child(folium.Element(legend_html))

    st_folium(peta_rth, use_container_width=True, height=550)

# ------------------------------------------------------------------------------
# BAGIAN 7: EKSPOR ARTIFAK DATA PENELITIAN
# ------------------------------------------------------------------------------
st.header("💾 Unduh Hasil")

kolom_export = [
    "nama_kecamatan", "latitude", "longitude", "luas_kec_km2", "luas_rth_km2",
    "persentase_rth", "Cluster_ID", "Status_Zona",
]
csv_buffer = StringIO()
df_final[kolom_export].to_csv(csv_buffer, index=False)

st.download_button(
    label="⬇️ Unduh Hasil_Clustering_RTH_Final.csv",
    data=csv_buffer.getvalue(),
    file_name="Hasil_Clustering_RTH_Final.csv",
    mime="text/csv",
)

peta_html = peta_rth.get_root().render()
st.download_button(
    label="⬇️ Unduh Peta_Zonasi_RTH_Sukabumi.html",
    data=peta_html,
    file_name="Peta_Zonasi_RTH_Sukabumi.html",
    mime="text/html",
)

st.caption(
    "Aplikasi ini merupakan terjemahan Python/Streamlit dari script R `RTH.R` "
    "(Program Studi Teknik Informatika, Universitas Muhammadiyah Sukabumi)."
)

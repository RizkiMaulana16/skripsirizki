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

st.title("🌳 Klasifikasi Zona Ruang Terbuka Hijau (RTH)")
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

# Pemetaan umum Cluster_ID -> label zona (dipakai lagi di fitur klasifikasi data baru)
cluster_to_label = df_final.drop_duplicates("Cluster_ID").set_index("Cluster_ID")["Status_Zona"].to_dict()

# ------------------------------------------------------------------------------
# BAGIAN 5B: TABEL HASIL KLASIFIKASI PER KECAMATAN
# ------------------------------------------------------------------------------
st.header("📋 Hasil Klasifikasi per Kecamatan")
st.markdown(
    "Tabel berikut menampilkan **klaster (Cluster_ID) dan status zona** "
    "untuk setiap kecamatan berdasarkan hasil K-Means di atas."
)

# Ringkasan jumlah kecamatan per zona/klaster
ringkasan_zona = (
    df_final.groupby("Status_Zona")["nama_kecamatan"]
    .count()
    .reset_index()
    .rename(columns={"nama_kecamatan": "Jumlah Kecamatan"})
    .sort_values("Jumlah Kecamatan", ascending=False)
)
col_ringkas1, col_ringkas2 = st.columns([1, 2])
with col_ringkas1:
    st.dataframe(ringkasan_zona, use_container_width=True, hide_index=True)
with col_ringkas2:
    fig_ringkas = px.bar(
        ringkasan_zona,
        x="Status_Zona",
        y="Jumlah Kecamatan",
        color="Status_Zona",
        text="Jumlah Kecamatan",
        title="Jumlah Kecamatan per Zona/Klaster",
    )
    fig_ringkas.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah Kecamatan")
    st.plotly_chart(fig_ringkas, use_container_width=True)

# Filter interaktif berdasarkan zona
opsi_zona = ["Semua Zona"] + sorted(df_final["Status_Zona"].unique().tolist())
filter_zona = st.selectbox("Filter berdasarkan Status Zona:", opsi_zona)

tabel_klasifikasi = df_final[
    ["nama_kecamatan", "Cluster_ID", "Status_Zona", "luas_kec_km2", "luas_rth_km2", "persentase_rth"]
].rename(
    columns={
        "nama_kecamatan": "Nama Kecamatan",
        "Cluster_ID": "Klaster",
        "Status_Zona": "Status Zona",
        "luas_kec_km2": "Luas Kecamatan (km²)",
        "luas_rth_km2": "Luas RTH (km²)",
        "persentase_rth": "Persentase RTH (%)",
    }
).sort_values("Klaster")

if filter_zona != "Semua Zona":
    tabel_klasifikasi = tabel_klasifikasi[tabel_klasifikasi["Status Zona"] == filter_zona]

st.dataframe(
    tabel_klasifikasi.round(2),
    use_container_width=True,
    hide_index=True,
    height=400,
)

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

    # Legenda dibuat sebagai Leaflet Control asli (L.control), lalu di-addTo() langsung
    # ke objek peta. Ini cara resmi Leaflet untuk elemen pojok peta, sehingga posisinya
    # selalu benar dan tidak terpotong -- berbeda dengan <div> biasa yang ditempel di body
    # halaman, yang gampang salah posisi karena Leaflet memakai CSS transform untuk pan/zoom.
    from branca.element import MacroElement, Template

    legend_template = """
    {% macro script(this, kwargs) %}
    var legend_{{ this.get_name() }} = L.control({position: 'bottomright'});
    legend_{{ this.get_name() }}.onAdd = function (map) {
        var div = L.DomUtil.create('div', 'info legend');
        div.style.background = 'white';
        div.style.padding = '10px 14px';
        div.style.border = '1px solid grey';
        div.style.borderRadius = '5px';
        div.style.width = '190px';
        div.style.boxShadow = '2px 2px 6px rgba(0,0,0,0.3)';
        div.style.fontSize = '13px';
        div.style.fontFamily = 'Arial, sans-serif';
        div.style.whiteSpace = 'nowrap';
        div.innerHTML =
            '<b>Legenda Zona RTH</b><br>' +
            '<i style="background:green;width:10px;height:10px;display:inline-block;margin-right:6px;"></i>Zona RTH Tinggi<br>' +
            '<i style="background:orange;width:10px;height:10px;display:inline-block;margin-right:6px;"></i>Zona RTH Sedang<br>' +
            '<i style="background:red;width:10px;height:10px;display:inline-block;margin-right:6px;"></i>Zona RTH Rendah';
        return div;
    };
    legend_{{ this.get_name() }}.addTo({{ this._parent.get_name() }});
    {% endmacro %}
    """
    legend = MacroElement()
    legend._template = Template(legend_template)
    peta_rth.add_child(legend)

    st_folium(peta_rth, use_container_width=True, height=550)

# ------------------------------------------------------------------------------
# BAGIAN 6B: FITUR KLASIFIKASI DATA / WILAYAH BARU
# ------------------------------------------------------------------------------
st.header("🔎 Klasifikasikan Kecamatan/Wilayah Baru")
st.markdown(
    "Masukkan data suatu wilayah untuk memprediksi masuk **klaster & zona RTH mana** "
    "berdasarkan model K-Means yang sudah dilatih di atas (K = "
    f"**{k_optimal}**)."
)

# Batas nilai asli (sebelum normalisasi) dari data training -> dipakai untuk
# menormalisasi input baru dengan skala Min-Max yang SAMA seperti saat training.
batas_min = data_model.min()
batas_max = data_model.max()

def klasifikasikan_wilayah(luas_kec_km2_baru: float, luas_rth_km2_baru: float):
    """Menghitung persentase RTH, menormalisasi dgn skala training, lalu memprediksi klaster."""
    persentase_baru = (luas_rth_km2_baru / luas_kec_km2_baru) * 100

    norm_luas = (luas_kec_km2_baru - batas_min["luas_kec_km2"]) / (
        batas_max["luas_kec_km2"] - batas_min["luas_kec_km2"]
    )
    norm_persen = (persentase_baru - batas_min["persentase_rth"]) / (
        batas_max["persentase_rth"] - batas_min["persentase_rth"]
    )

    ekstrapolasi = not (0 <= norm_luas <= 1 and 0 <= norm_persen <= 1)
    # Clamp ke rentang 0-1 supaya prediksi tetap masuk akal walau data di luar
    # rentang training (ekstrapolasi Min-Max)
    norm_luas_clamped = min(max(norm_luas, 0), 1)
    norm_persen_clamped = min(max(norm_persen, 0), 1)

    cluster_pred = model_kmeans.predict([[norm_luas_clamped, norm_persen_clamped]])[0] + 1
    label_pred = cluster_to_label.get(cluster_pred, f"Klaster {cluster_pred}")

    return {
        "persentase_rth": persentase_baru,
        "cluster": cluster_pred,
        "label": label_pred,
        "ekstrapolasi": ekstrapolasi,
    }

tab_manual, tab_batch = st.tabs(["✍️ Input Manual", "📁 Upload CSV (Banyak Wilayah)"])

with tab_manual:
    with st.form("form_klasifikasi_manual"):
        c1, c2, c3 = st.columns(3)
        with c1:
            nama_wilayah_baru = st.text_input("Nama Wilayah (opsional)", value="Wilayah Baru")
        with c2:
            luas_kec_input = st.number_input(
                "Luas Kecamatan (km²)", min_value=0.01, value=50.0, step=1.0, format="%.2f"
            )
        with c3:
            luas_rth_input = st.number_input(
                "Luas Ruang Terbuka Hijau (km²)", min_value=0.0, value=25.0, step=1.0, format="%.2f"
            )
        submit_manual = st.form_submit_button("🔍 Klasifikasikan", use_container_width=True)

    if submit_manual:
        hasil = klasifikasikan_wilayah(luas_kec_input, luas_rth_input)

        warna_hasil = {
            "Zona RTH Tinggi (Sangat Baik)": "success",
            "Zona RTH Sedang (Cukup/Ideal)": "warning",
            "Zona RTH Rendah (Kritis)": "error",
        }.get(hasil["label"], "info")

        colA, colB, colC = st.columns(3)
        colA.metric("Persentase RTH", f"{hasil['persentase_rth']:.2f} %")
        colB.metric("Klaster (Cluster_ID)", int(hasil["cluster"]))
        colC.metric("Status Zona", hasil["label"])

        getattr(st, warna_hasil)(
            f"**{nama_wilayah_baru}** diklasifikasikan ke dalam **Klaster {hasil['cluster']}** "
            f"— **{hasil['label']}**."
        )
        if hasil["ekstrapolasi"]:
            st.warning(
                "⚠️ Nilai input berada di luar rentang data training (ekstrapolasi), "
                "sehingga hasil normalisasi dibatasi (clamp) ke rentang 0-1. "
                "Prediksi tetap dihasilkan, namun tingkat keandalannya lebih rendah "
                "dibanding wilayah yang nilainya masih dalam rentang data asli."
            )

with tab_batch:
    st.markdown(
        "Unggah CSV berisi kolom **`nama_wilayah`, `luas_kec_km2`, `luas_rth_km2`** "
        "untuk mengklasifikasikan banyak wilayah baru sekaligus."
    )
    contoh_csv = pd.DataFrame(
        {"nama_wilayah": ["Contoh A", "Contoh B"], "luas_kec_km2": [45.2, 60.0], "luas_rth_km2": [20.1, 15.5]}
    )
    st.download_button(
        "⬇️ Unduh Template CSV",
        data=contoh_csv.to_csv(index=False),
        file_name="template_klasifikasi_baru.csv",
        mime="text/csv",
    )

    file_batch = st.file_uploader("Unggah file CSV wilayah baru", type=["csv"], key="batch_klasifikasi")
    if file_batch is not None:
        try:
            df_batch = pd.read_csv(file_batch)
            hasil_batch = df_batch.apply(
                lambda r: pd.Series(klasifikasikan_wilayah(r["luas_kec_km2"], r["luas_rth_km2"])),
                axis=1,
            )
            df_batch_hasil = pd.concat([df_batch, hasil_batch], axis=1).rename(
                columns={
                    "persentase_rth": "Persentase RTH (%)",
                    "cluster": "Klaster",
                    "label": "Status Zona",
                    "ekstrapolasi": "Ekstrapolasi?",
                }
            )
            st.dataframe(df_batch_hasil.round(2), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Unduh Hasil Klasifikasi Batch",
                data=df_batch_hasil.to_csv(index=False),
                file_name="Hasil_Klasifikasi_Wilayah_Baru.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(
                f"Gagal memproses file: {e}. Pastikan kolom `nama_wilayah`, "
                "`luas_kec_km2`, `luas_rth_km2` tersedia."
            )

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

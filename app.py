import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime, timezone, timedelta
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# Library tambahan untuk auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ---------------------------
# Konfigurasi Proyek
# ---------------------------
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/class/session5/dht"
TOPIC_OUTPUT = "iot/class/session5/led"
MODEL_PATH = "iot_temp_model.pkl"
ANOMALY_THRESHOLD = 0.20

# Helper untuk Timezone GMT+7
TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# State Initialization
# ---------------------------
if "mqtt_client" not in st.session_state:
    st.session_state["mqtt_client"] = None
    st.session_state["is_connected"] = False
    
if "data_buffer" not in st.session_state:
    st.session_state["data_buffer"] = []
    
if "log_df" not in st.session_state:
    st.session_state["log_df"] = pd.DataFrame(columns=["ts", "temp", "hum", "pred", "proba", "anomaly", "score", "alert_status"])

# NEW: State untuk Manual Override
if "manual_mode_active" not in st.session_state:
    st.session_state["manual_mode_active"] = False
if "manual_alert_status" not in st.session_state:
    st.session_state["manual_alert_status"] = "ALERT_OFF"
if "latest_ml_alert" not in st.session_state:
    st.session_state["latest_ml_alert"] = "ALERT_OFF"


# ---------------------------
# Inisialisasi Model
# ---------------------------
@st.cache_resource
def load_ml_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if "model_loaded" not in st.session_state:
    st.session_state["model"] = load_ml_model(MODEL_PATH)
    st.session_state["model_loaded"] = st.session_state["model"] is not None


# ---------------------------\
# Fungsi Callback MQTT
# ---------------------------\
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)
        st.session_state["is_connected"] = True
        print("MQTT Connected and Subscribed.")
    else:
        st.session_state["is_connected"] = False
        print(f"MQTT Connection Failed: {rc}")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data.get("temp", np.nan))
        hum = float(data.get("hum", np.nan))
        ts = now_str()

        # JANGAN PUBLISH DI SINI, HANYA HITUNG DAN SIMPAN KE STATE
        alert_status_ml = "ALERT_OFF"
        max_proba = np.nan
        pred_class = "N/A"
        anomaly_score = np.nan
        is_anomaly = False

        if st.session_state.get("model_loaded", False) and not np.isnan(temp):
            # 1. Prediksi ML
            X = np.array([[temp, hum]])
            pred_class = st.session_state["model"].predict(X)[0]
            proba = st.session_state["model"].predict_proba(X)[0]
            max_proba = np.max(proba)
            
            # Deteksi Anomali
            anomaly_score = 1.0 - max_proba
            is_anomaly = anomaly_score > ANOMALY_THRESHOLD
            
            # Tentukan ML Alert Status
            if pred_class == "Panas" or is_anomaly: 
                alert_status_ml = "ALERT_ON"
        
        # Simpan hasil prediksi ML terbaru
        st.session_state["latest_ml_alert"] = alert_status_ml
        
        # 2. Masukkan Data ke buffer
        new_data = {
            "ts": ts, "temp": temp, "hum": hum, "pred": pred_class, 
            "proba": max_proba, "anomaly": is_anomaly, "score": anomaly_score,
            "alert_status": alert_status_ml # Simpan status ML sebelum Override
        }
        st.session_state["data_buffer"].append(new_data)
            
    except Exception as e:
        print(f"Error processing message: {e}")

# ---------------------------\
# Fungsi Kontrol & Publikasi (Dipanggil setiap RERUN)
# ---------------------------\
def init_mqtt_client():
    if st.session_state["mqtt_client"] is None:
        client = mqtt.Client(client_id=f"StreamlitClient-{time.time()}")
        client.on_connect = on_connect
        client.on_message = on_message
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            st.session_state["mqtt_client"] = client
        except Exception as e:
            st.session_state["is_connected"] = False
            st.session_state["mqtt_client"] = None
            print(f"Initial MQTT connection failed: {e}")

def process_buffer_and_update_ui():
    client = st.session_state.get("mqtt_client")
    if client:
        # PENTING: client.loop() dipanggil secara non-blocking setiap 500ms refresh
        client.loop(timeout=0.01) 
        
    # Ambil data dari buffer dan masukkan ke DataFrame
    if st.session_state["data_buffer"]:
        new_data_list = st.session_state["data_buffer"]
        st.session_state["data_buffer"] = [] # Bersihkan buffer

        new_df = pd.DataFrame(new_data_list)
        st.session_state["log_df"] = pd.concat([st.session_state["log_df"], new_df], ignore_index=True)
        if len(st.session_state["log_df"]) > 100:
            st.session_state["log_df"] = st.session_state["log_df"].iloc[-100:]

def publish_output_control():
    client = st.session_state.get("mqtt_client")
    if not client or not st.session_state.get("is_connected"):
        return

    # Tentukan status akhir: Manual Override > ML Prediction
    if st.session_state["manual_mode_active"]:
        final_alert = st.session_state["manual_alert_status"]
    else:
        final_alert = st.session_state["latest_ml_alert"]

    # Kirim ke ESP32
    client.publish(TOPIC_OUTPUT, final_alert, qos=0)
    
    # Update status di metrik terbaru (jika ada data)
    if not st.session_state["log_df"].empty:
        # Update kolom alert_status pada baris terakhir dengan status yang benar-benar dikirim
        st.session_state["log_df"].loc[st.session_state["log_df"].index[-1], "alert_status"] = final_alert


# --- Panggil koneksi dan proses saat pertama kali ---
init_mqtt_client()
process_buffer_and_update_ui()
publish_output_control() # Panggil kontrol di akhir loop

# ---------------------------\
# STREAMLIT UI Rendering
# ---------------------------\
st.set_page_config(layout="wide", page_title="IoT Temperature Monitoring & Control")

# Sidebar
st.sidebar.title("Configuration")

# Status Koneksi
if st.session_state.get("is_connected"):
    st.sidebar.success("MQTT Connected!")
else:
    st.sidebar.error("MQTT Disconnected!")

st.sidebar.markdown("---")
st.sidebar.header("Kontrol Manual (Override)")
current_control = "MANUAL" if st.session_state["manual_mode_active"] else "OTOMATIS (ML)"
st.sidebar.markdown(f"**Mode Kontrol Saat Ini:** **`{current_control}`**")

# Logika Tombol Manual Override
col_man1, col_man2 = st.sidebar.columns(2)

if col_man1.button("🔥 Force ALERT ON", use_container_width=True):
    st.session_state["manual_mode_active"] = True
    st.session_state["manual_alert_status"] = "ALERT_ON"
    st.rerun() # PENTING: Force rerun agar publish_output_control dipanggil

if col_man2.button("🟢 Kembali ke AUTO", use_container_width=True):
    st.session_state["manual_mode_active"] = False
    st.session_state["manual_alert_status"] = "ALERT_OFF" # Reset status target manual
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.metric("Anomaly Threshold", f"{ANOMALY_THRESHOLD*100:.0f}% Score")
st.sidebar.metric("Model Status", "Loaded" if st.session_state.get("model_loaded") else "Failed")


st.title("🌡️ Real-Time IoT Temperature Monitoring & Control")

# Auto-refresh mechanism (500ms untuk Polling Data)
if HAS_AUTOREFRESH:
    st_autorefresh(interval=500, key="data_refresh_trigger")

# Metrics Display
if not st.session_state["log_df"].empty:
    latest_data = st.session_state["log_df"].iloc[-1]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Temperature (°C)", f"{latest_data['temp']:.2f}")
    col2.metric("Humidity (%)", f"{latest_data['hum']:.2f}")
    
    # Status Prediksi ML
    status_emoji = "🔥" if latest_data['pred'] == "Panas" else "🟢" if latest_data['pred'] == "Normal" else "❄️"
    col3.metric("ML Status", f"{status_emoji} {latest_data['pred']}", delta=f"Confidence: {latest_data['proba']*100:.1f}%")

    # Status Anomali
    anomaly_status = "⚠️ ANOMALI" if latest_data['anomaly'] else "OK"
    col4.metric("Data Anomaly", anomaly_status, delta=f"Score: {latest_data['score']:.2f}")
    
    # Status Aksi LED (Manual/Otomatis)
    led_status = latest_data.get("alert_status", "N/A")
    color = "red" if led_status == "ALERT_ON" else "green"
    
    col5.markdown(f"""
        <div style="background-color:{color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
            <p style="margin: 0; font-size: 14px;">ACTUATOR STATUS</p>
            <h3 style="margin: 0; font-weight: bold;">{led_status}</h3>
        </div>
    """, unsafe_allow_html=True)
    
# Plotting
st.markdown("### Real-Time Trend")
df_plot = st.session_state["log_df"].copy()
if not df_plot.empty:
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"], mode="lines+markers", name="Temp (°C)"))
    fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], mode="lines+markers", name="Hum (%)", yaxis="y2"))
    fig.update_layout(
        xaxis=dict(tickformat="%H:%M:%S"),
        yaxis=dict(title="Temp (°C)"),
        yaxis2=dict(title="Humidity (%)", overlaying="y", side="right", showgrid=False),
        height=520
    )

    colors = []
    for _, r in df_plot.iterrows():
        # Visualisasi menggunakan alert status yang benar-benar dikirim (final_alert)
        status = r.get("alert_status", r.get("latest_ml_alert", "ALERT_OFF"))
        if r.get("anomaly"):
            colors.append("magenta")
        elif status == "ALERT_ON": 
            colors.append("red")
        elif r.get("pred", "") == "Normal":
            colors.append("green")
        else:
            colors.append("blue")
            
    fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="lines+markers"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data yet. Make sure ESP32 publishes to correct topic.")

st.markdown("---")
col_log, col_export = st.columns([3, 1])

with col_log:
    col_log.markdown("### Recent Logs")
    if not st.session_state["log_df"].empty:
        st.dataframe(st.session_state["log_df"].tail(10), use_container_width=True)

with col_export:
    st.markdown("### Export Data")
    if st.session_state["log_df"].empty:
        st.warning("No data to export.")
    else:
        # 2. Implementasi Export Data (Download Button) [cite: 465, 444]
        csv_export = st.session_state["log_df"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Full Log Data (.csv)",
            data=csv_export,
            file_name=f'iot_log_{datetime.now(TZ).strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            use_container_width=True
        )

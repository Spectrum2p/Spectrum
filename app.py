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
# State Initialization & Fungsi Non-UI (Tidak Berubah)
# ---------------------------
if "mqtt_client" not in st.session_state:
    st.session_state["mqtt_client"] = None
    st.session_state["is_connected"] = False
    
if "data_buffer" not in st.session_state:
    st.session_state["data_buffer"] = []
    
if "log_df" not in st.session_state:
    st.session_state["log_df"] = pd.DataFrame(columns=["ts", "temp", "hum", "pred", "proba", "anomaly", "score", "alert_status"])

if "manual_mode_active" not in st.session_state:
    st.session_state["manual_mode_active"] = False
if "manual_alert_status" not in st.session_state:
    st.session_state["manual_alert_status"] = "ALERT_OFF"
if "latest_ml_alert" not in st.session_state:
    st.session_state["latest_ml_alert"] = "ALERT_OFF"

@st.cache_resource
def load_ml_model(path):
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return None

if "model_loaded" not in st.session_state:
    st.session_state["model"] = load_ml_model(MODEL_PATH)
    st.session_state["model_loaded"] = st.session_state["model"] is not None

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)
        st.session_state["is_connected"] = True
    else:
        st.session_state["is_connected"] = False

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data.get("temp", np.nan))
        hum = float(data.get("hum", np.nan))
        ts = now_str()
        alert_status_ml = "ALERT_OFF"
        max_proba = np.nan
        pred_class = "N/A"
        anomaly_score = np.nan
        is_anomaly = False

        if st.session_state.get("model_loaded", False) and not np.isnan(temp):
            X = np.array([[temp, hum]])
            pred_class = st.session_state["model"].predict(X)[0]
            proba = st.session_state["model"].predict_proba(X)[0]
            max_proba = np.max(proba)
            
            anomaly_score = 1.0 - max_proba
            is_anomaly = anomaly_score > ANOMALY_THRESHOLD
            
            if pred_class == "Panas" or is_anomaly: 
                alert_status_ml = "ALERT_ON"
        
        st.session_state["latest_ml_alert"] = alert_status_ml
        
        new_data = {
            "ts": ts, "temp": temp, "hum": hum, "pred": pred_class, 
            "proba": max_proba, "anomaly": is_anomaly, "score": anomaly_score,
            "alert_status": alert_status_ml
        }
        st.session_state["data_buffer"].append(new_data)
            
    except Exception:
        pass

def init_mqtt_client():
    if st.session_state["mqtt_client"] is None:
        client = mqtt.Client(client_id=f"StreamlitClient-{time.time()}")
        client.on_connect = on_connect
        client.on_message = on_message
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            st.session_state["mqtt_client"] = client
        except Exception:
            st.session_state["is_connected"] = False
            st.session_state["mqtt_client"] = None

def process_buffer_and_update_ui():
    client = st.session_state.get("mqtt_client")
    if client:
        client.loop(timeout=0.01) 
        
    if st.session_state["data_buffer"]:
        new_data_list = st.session_state["data_buffer"]
        st.session_state["data_buffer"] = []

        new_df = pd.DataFrame(new_data_list)
        st.session_state["log_df"] = pd.concat([st.session_state["log_df"], new_df], ignore_index=True)
        if len(st.session_state["log_df"]) > 100:
            st.session_state["log_df"] = st.session_state["log_df"].iloc[-100:]

def publish_output_control():
    client = st.session_state.get("mqtt_client")
    if not client or not st.session_state.get("is_connected"):
        return

    if st.session_state["manual_mode_active"]:
        final_alert = st.session_state["manual_alert_status"]
    else:
        final_alert = st.session_state["latest_ml_alert"]

    client.publish(TOPIC_OUTPUT, final_alert, qos=0)
    
    if not st.session_state["log_df"].empty:
        st.session_state["log_df"].loc[st.session_state["log_df"].index[-1], "alert_status"] = final_alert

# --- Panggil koneksi dan proses saat pertama kali ---
init_mqtt_client()
process_buffer_and_update_ui()
publish_output_control()

# ---------------------------\
# STREAMLIT UI Rendering
# ---------------------------\
st.set_page_config(layout="wide", page_title="IoT Temperature Monitoring & Control")

# ... (Sidebar Code - No Change)
st.sidebar.title("Configuration")
if st.session_state.get("is_connected"):
    st.sidebar.success("MQTT Connected!")
else:
    st.sidebar.error("MQTT Disconnected!")
st.sidebar.markdown("---")
st.sidebar.header("Kontrol Manual (Override)")
current_control = "MANUAL" if st.session_state["manual_mode_active"] else "OTOMATIS (ML)"
st.sidebar.markdown(f"**Mode Kontrol Saat Ini:** **`{current_control}`**")
col_man1, col_man2 = st.sidebar.columns(2)
if col_man1.button("🔥 Force ALERT ON", use_container_width=True):
    st.session_state["manual_mode_active"] = True
    st.session_state["manual_alert_status"] = "ALERT_ON"
    st.rerun()
if col_man2.button("🟢 Kembali ke AUTO", use_container_width=True):
    st.session_state["manual_mode_active"] = False
    st.session_state["manual_alert_status"] = "ALERT_OFF"
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.metric("Anomaly Threshold", f"{ANOMALY_THRESHOLD*100:.0f}% Score")
st.sidebar.metric("Model Status", "Loaded" if st.session_state.get("model_loaded") else "Failed")


st.title("🌡️ Real-Time IoT Temperature Monitoring & Control")
if HAS_AUTOREFRESH:
    st_autorefresh(interval=500, key="data_refresh_trigger")

# Metrics Display (Menggunakan custom HTML/Markdown untuk kontras)
if not st.session_state["log_df"].empty:
    latest_data = st.session_state["log_df"].iloc[-1]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # KOREKSI 1: TEMPERATURE (Warna Biru)
    col1.markdown(f"""
        <div style="background-color: lightgray; padding: 10px; border-radius: 5px; color: black; text-align: center; border-left: 5px solid blue;">
            <p style="margin: 0; font-size: 14px; font-weight: bold;">TEMPERATURE (°C)</p>
            <h3 style="margin: 0; font-weight: bold; color: blue;">{latest_data['temp']:.2f}</h3>
        </div>
    """, unsafe_allow_html=True)

    # KOREKSI 2: HUMIDITY (Warna Hijau)
    col2.markdown(f"""
        <div style="background-color: lightgray; padding: 10px; border-radius: 5px; color: black; text-align: center; border-left: 5px solid green;">
            <p style="margin: 0; font-size: 14px; font-weight: bold;">HUMIDITY (%)</p>
            <h3 style="margin: 0; font-weight: bold; color: green;">{latest_data['hum']:.2f}</h3>
        </div>
    """, unsafe_allow_html=True)

    # ML Status (Warna Merah/Biru/Hijau - Tidak Berubah)
    pred = latest_data['pred']
    proba = latest_data['proba']
    if pred == "Panas":
        color = "red"
        emoji = "🔥"
    elif pred == "Dingin":
        color = "blue"
        emoji = "❄️"
    else:
        color = "green"
        emoji = "🟢"

    col3.markdown(f"""
        <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
            <p style="margin: 0; font-size: 14px;">ML STATUS</p>
            <h3 style="margin: 0; font-weight: bold;">{emoji} {pred}</h3>
            <p style="margin: 0; font-size: 12px;">Confidence: {proba*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

    # Status Anomali (Tidak Berubah)
    anomaly_status = "⚠️ ANOMALI" if latest_data['anomaly'] else "OK"
    anomaly_bg_color = "yellow" if latest_data['anomaly'] else "lightgray"
    anomaly_text_color = "black" if latest_data['anomaly'] else "green"
    
    col4.markdown(f"""
        <div style="background-color: {anomaly_bg_color}; padding: 10px; border-radius: 5px; color: {anomaly_text_color}; text-align: center; border-left: 5px solid {anomaly_text_color};">
            <p style="margin: 0; font-size: 14px; font-weight: bold;">DATA ANOMALY</p>
            <h3 style="margin: 0; font-weight: bold;">{anomaly_status}</h3>
            <p style="margin: 0; font-size: 12px;">Score: {latest_data['score']:.2f}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Status Aksi LED (Actuator) (Tidak Berubah)
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
    
    # KOREKSI 3: Garis Suhu (BLUE) dan Marker Warna SAMA (Blue)
    fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"], 
                             mode="lines+markers", 
                             name="Temp (°C)",
                             line=dict(color='blue', width=3),
                             marker=dict(size=10, color='blue', line=dict(width=1, color='black')))) 
    
    # KOREKSI 4: Garis Kelembaban (GREEN) dan Marker Warna SAMA (Green)
    fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], 
                             mode="lines+markers", 
                             name="Hum (%)", 
                             # yaxis2 dihapus untuk single axis
                             line=dict(color='green', width=3),
                             marker=dict(size=10, color='green', line=dict(width=1, color='black')))) 
                             
    fig.update_layout(
        xaxis=dict(tickformat="%H:%M:%S"),
        # KOREKSI 5: Single Y-Axis (1-100 scale)
        yaxis=dict(title="Skala Gabungan (0-100)", 
                   showgrid=True,
                   range=[0, 100]), 
        height=520,
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )

    # Note: Logika pewarnaan marker berdasarkan prediksi ML dihapus dan diganti dengan warna garis (blue/green)
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data yet. Make sure ESP32 publishes to correct topic.")

st.markdown("---")
col_log, col_export = st.columns([3, 1])

with col_log:
    col_log.markdown("### Recent Logs")
    if not st.session_state["log_df"].empty:
        # Tabel Log Lengkap dan Terurut
        df_log = st.session_state["log_df"].tail(10).copy()
        
        column_order = ["ts", "temp", "hum", "pred", "alert_status", "anomaly", "proba", "score"]
        column_names = {
            "ts": "Timestamp",
            "temp": "Temp (°C)",
            "hum": "Hum (%)",
            "pred": "Prediction",
            "alert_status": "Alert Status",
            "anomaly": "Anomaly",
            "proba": "Confidence",
            "score": "Anomaly Score"
        }
        
        df_display = df_log[column_order].rename(columns=column_names)
        
        st.dataframe(df_display, use_container_width=True)

with col_export:
    st.markdown("### Export Data")
    if st.session_state["log_df"].empty:
        st.warning("No data to export.")
    else:
        csv_export = st.session_state["log_df"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Full Log Data (.csv)",
            data=csv_export,
            file_name=f'iot_log_{datetime.now(TZ).strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            use_container_width=True
        )

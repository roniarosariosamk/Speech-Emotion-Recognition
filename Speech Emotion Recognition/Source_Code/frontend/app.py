import io
import requests
import streamlit as st
import pandas as pd
import soundfile as sf


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Speech Emotion AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/predict"


# ==========================================
# EMOTION CONFIGURATION
# ==========================================

EMOTION_DETAILS = {
    "neutral": ("😐", "#94A3B8"),
    "calm": ("😌", "#38BDF8"),
    "happy": ("😊", "#FACC15"),
    "sad": ("😢", "#60A5FA"),
    "angry": ("😡", "#F87171"),
    "fearful": ("😨", "#A78BFA"),
    "disgust": ("🤢", "#4ADE80"),
    "surprised": ("😲", "#FB923C")
}


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080D1B, #111827);
    color: #F8FAFC;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, #312E81, #4F46E5, #7C3AED);
    padding: 38px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 35px rgba(79,70,229,0.25);
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #E0E7FF;
    font-size: 16px;
}

.section-title {
    color: #A5B4FC;
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.result-card {
    background: linear-gradient(135deg, #1E293B, #312E81);
    border: 1px solid #6366F1;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 15px;
}

.result-card h2 {
    color: white;
    font-size: 34px;
    margin: 10px 0;
}

.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

.metric-card h3 {
    color: #A5B4FC;
    font-size: 15px;
}

.metric-card p {
    color: white;
    font-size: 23px;
    font-weight: bold;
}

div.stButton > button {
    background: linear-gradient(90deg, #4F46E5, #9333EA);
    color: white;
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 17px;
    font-weight: bold;
}

div.stButton > button:hover {
    border: 1px solid #A5B4FC;
    color: white;
}

[data-testid="stFileUploader"] {
    background: #1E293B;
    padding: 20px;
    border-radius: 15px;
    border: 1px dashed #6366F1;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🎙️ Speech Emotion Recognition AI</h1>
    <p>
        Discover emotions hidden in human speech using
        MFCC features and Deep Learning.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🎧 About the AI")

    st.write(
        "An intelligent speech analysis application "
        "that predicts emotions from uploaded audio."
    )

    st.divider()

    st.markdown("### 🧠 Model Pipeline")

    st.write("1. Audio Upload")
    st.write("2. Audio Preprocessing")
    st.write("3. MFCC Extraction")
    st.write("4. Deep Learning")
    st.write("5. Emotion Prediction")

    st.divider()

    st.markdown("### 🎭 Supported Emotions")

    for emotion, details in EMOTION_DETAILS.items():
        st.write(f"{details[0]} {emotion.capitalize()}")

    st.divider()

    st.caption("Academic Machine Learning Project")


# ==========================================
# AUDIO UPLOAD
# ==========================================

st.markdown(
    '<div class="section-title">📁 Upload Your Audio</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Drag and drop your audio file here",
    type=["wav", "mp3", "flac", "ogg"]
)


# ==========================================
# AUDIO PLAYBACK
# ==========================================

if uploaded_file is not None:

    st.success(f"Audio uploaded: {uploaded_file.name}")

    st.markdown("### ▶️ Audio Preview")

    try:

        audio_data, sample_rate = sf.read(
            io.BytesIO(uploaded_file.getvalue()),
            dtype="float32"
        )

        duration = len(audio_data) / sample_rate

        audio_buffer = io.BytesIO()

        sf.write(
            audio_buffer,
            audio_data,
            sample_rate,
            format="WAV",
            subtype="PCM_16"
        )

        audio_buffer.seek(0)

        st.audio(audio_buffer, format="audio/wav")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Audio Duration</h3>
                    <p>{duration:.2f} sec</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Sample Rate</h3>
                    <p>{sample_rate} Hz</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as error:
        st.error(f"Audio preview error: {error}")


    st.divider()


    # ======================================
    # EMOTION PREDICTION
    # ======================================

    if st.button("🚀 ANALYZE MY EMOTION", use_container_width=True):

        with st.spinner("🧠 AI is analyzing your speech..."):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream"
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=180
                )

                response.raise_for_status()

                result = response.json()

                predicted_emotion = result["predicted_emotion"].lower()
                probabilities = result["probabilities"]

                st.session_state["result"] = result

            except requests.exceptions.ConnectionError:

                st.error(
                    "Backend connection failed. "
                    "Please check whether FastAPI is running."
                )

            except requests.exceptions.Timeout:

                st.error("Prediction timed out. Please try again.")

            except requests.exceptions.HTTPError as error:

                st.error(f"Backend error: {error}")

            except Exception as error:

                st.error(f"Something went wrong: {error}")


# ==========================================
# DISPLAY RESULTS
# ==========================================

if "result" in st.session_state:

    result = st.session_state["result"]

    predicted_emotion = result["predicted_emotion"].lower()
    probabilities = result["probabilities"]

    emoji, emotion_color = EMOTION_DETAILS.get(
        predicted_emotion,
        ("🎭", "#A5B4FC")
    )

    st.divider()

    st.markdown(
        '<div class="section-title">🧠 AI Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div style="font-size:65px;">{emoji}</div>
            <p style="color:#C7D2FE;">DETECTED EMOTION</p>
            <h2 style="color:{emotion_color};">
                {predicted_emotion.upper()}
            </h2>
            <p style="color:#CBD5E1;">
                AI-powered speech emotion classification
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ======================================
    # ALL EMOTION SCORES
    # ======================================

    st.markdown(
        '<div class="section-title">📊 Emotion Analysis</div>',
        unsafe_allow_html=True
    )

    scores_df = pd.DataFrame({
        "Emotion": list(probabilities.keys()),
        "Score": [
            float(value) * 100
            for value in probabilities.values()
        ]
    })

    scores_df = scores_df.sort_values(
        "Score",
        ascending=False
    )

    st.bar_chart(
        scores_df.set_index("Emotion"),
        horizontal=True,
        x_label="Emotion",
        y_label="Model Score (%)"
    )

    st.markdown("### 📋 Detailed Emotion Scores")

    display_df = scores_df.copy()
    display_df["Score"] = display_df["Score"].map(
        lambda value: f"{value:.2f}%"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Note: These are model output scores, not calibrated "
        "confidence values. Predictions may vary with audio quality "
        "and recording conditions."
    )

    if st.button("🔄 Analyze Another Audio", use_container_width=True):

        del st.session_state["result"]
        st.rerun()


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:20px; color:#94A3B8;">
    <h4>🎙️ Speech Emotion Recognition AI</h4>
    <p>MFCC Features | Deep Learning | 8 Emotion Classes</p>
    <p>Built as an Academic Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)
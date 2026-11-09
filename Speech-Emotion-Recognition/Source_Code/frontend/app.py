import io
import hashlib
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
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

API_URL = "https://speech-emotion-api-s5nv.onrender.com/predict"


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

/* ==========================================
   GLOBAL APPLICATION THEME
========================================== */

.stApp {
    background: radial-gradient(
        circle at top right,
        #1E1B4B 0%,
        #0B1120 45%,
        #080D1B 100%
    );
    color: #F8FAFC;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ==========================================
   HERO HEADER
========================================== */

.hero {
    background: linear-gradient(
        120deg,
        #312E81,
        #4F46E5,
        #7C3AED
    );
    padding: 45px 30px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0 15px 45px rgba(79, 70, 229, 0.25);
    border: 1px solid rgba(165, 180, 252, 0.2);
}

.hero h1 {
    color: #FFFFFF;
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 12px;
}

.hero p {
    color: #E0E7FF;
    font-size: 17px;
    line-height: 1.7;
}

/* ==========================================
   SECTION HEADINGS
========================================== */

.section-title {
    color: #C4B5FD;
    font-size: 25px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 18px;
}

/* ==========================================
   PREDICTION RESULT CARD
========================================== */

.result-card {
    background: linear-gradient(
        135deg,
        #1E293B,
        #312E81
    );
    border: 1px solid #818CF8;
    border-radius: 22px;
    padding: 35px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 12px 35px rgba(99, 102, 241, 0.15);
}

.result-card h2 {
    color: #FFFFFF;
    font-size: 36px;
    font-weight: 800;
    margin: 12px 0;
}

/* ==========================================
   AUDIO METRIC CARDS
========================================== */

.metric-card {
    background: linear-gradient(
        145deg,
        #1E293B,
        #172033
    );
    border: 1px solid #475569;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: #818CF8;
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.12);
}

.metric-card h3 {
    color: #A5B4FC;
    font-size: 15px;
    font-weight: 600;
}

.metric-card p {
    color: #FFFFFF;
    font-size: 25px;
    font-weight: 800;
}

/* ==========================================
   BUTTON DESIGN
========================================== */

div.stButton > button {
    background: linear-gradient(
        90deg,
        #4F46E5,
        #7C3AED
    );
    color: #FFFFFF;
    border: 1px solid transparent;
    border-radius: 14px;
    height: 54px;
    font-size: 17px;
    font-weight: 750;
    transition: all 0.3s ease;
    box-shadow: 0 5px 18px rgba(79, 70, 229, 0.25);
}

div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #6366F1,
        #9333EA
    );
    border: 1px solid #C4B5FD;
    color: #FFFFFF;
    transform: translateY(-2px);
}

/* ==========================================
   FILE UPLOADER
========================================== */

[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.8);
    padding: 22px;
    border-radius: 18px;
    border: 1px dashed #818CF8;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #C4B5FD;
    background: rgba(49, 46, 129, 0.25);
}

/* ==========================================
   SIDEBAR
========================================== */

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #334155;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #C4B5FD;
}

/* ==========================================
   AUDIO PLAYER
========================================== */

audio {
    width: 100%;
    border-radius: 12px;
}

/* ==========================================
   DIVIDERS
========================================== */

hr {
    border-color: #334155;
    margin-top: 25px;
    margin-bottom: 25px;
}

/* ==========================================
   FOOTER
========================================== */

footer {
    visibility: hidden;
}

/* ==========================================
   RESPONSIVE DESIGN
========================================== */

@media (max-width: 768px) {

    .hero {
        padding: 30px 18px;
    }

    .hero h1 {
        font-size: 29px;
    }

    .hero p {
        font-size: 14px;
    }

    .section-title {
        font-size: 21px;
    }

    .result-card {
        padding: 22px;
    }

    .result-card h2 {
        font-size: 28px;
    }

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
# RESET OLD PREDICTION WHEN NEW AUDIO IS UPLOADED
# ==========================================

if "previous_audio_hash" not in st.session_state:
    st.session_state["previous_audio_hash"] = None

if uploaded_file is not None:
    audio_bytes = uploaded_file.getvalue()
    current_audio_hash = hashlib.md5(audio_bytes).hexdigest()

    if current_audio_hash != st.session_state["previous_audio_hash"]:
        st.session_state["result"] = None
        st.session_state["previous_audio_hash"] = current_audio_hash

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

        # Validate audio duration
        if duration < 1:
            st.error(
                "⚠️ Audio is too short! "
                "Please upload an audio recording of at least 1 second."
            )
            st.stop()

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

if st.session_state.get("result") is not None:

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

        # ======================================
    # PROFESSIONAL EMOTION ANALYSIS CHART
    # ======================================

    chart_df = scores_df.sort_values(
        by="Score",
        ascending=True
    )

    fig = px.bar(
        chart_df,
        x="Score",
        y="Emotion",
        orientation="h",
        text=chart_df["Score"].map(lambda x: f"{x:.2f}%")
    )

    fig.update_traces(
        marker_color="#818CF8",
        marker_line_color="#C4B5FD",
        marker_line_width=1,
        textposition="outside",
        textfont=dict(
            color="#FFFFFF",
            size=13
        ),
        cliponaxis=False
    )

    fig.update_layout(
        title="Emotion Score Distribution",
        xaxis=dict(
            title="Model Score (%)",
            range=[0, 100],
            ticksuffix="%",
            dtick=10,
            gridcolor="#334155",
            color="#E2E8F0"
        ),
        yaxis=dict(
            title="Emotion",
            color="#E2E8F0"
        ),
        plot_bgcolor="#0B1120",
        paper_bgcolor="#0B1120",
        font=dict(color="#FFFFFF"),
        height=430,
        margin=dict(l=20, r=50, t=70, b=50)
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
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

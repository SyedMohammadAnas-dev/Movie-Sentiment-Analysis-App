# venv/Scripts/activate
# cd sentiment-streamlit-app
# streamlit run app.py

# git add .python-version
# git commit -m "Specify Python version for Render"
# git push origin main
import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from streamlit_echarts import st_echarts
import pickle

# Load model and tokenizer
model = load_model("sentiment_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("🎬 Movie Sentiment Analysis App")
review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    # Preprocess input
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=200)

    # Predict
    prediction = model.predict(padded)
    confidence = float(prediction[0][0])

    # Decide label
    sentiment = "😃 Positive" if confidence > 0.5 else "😞 Negative"
    st.markdown(f"## Sentiment: {sentiment}")
    st.markdown(f"**Confidence Score:** {confidence:.2f}")

    # Prepare graph data
    pos_score = confidence
    neg_score = 1 - confidence

    option = {
        "title": {"text": "Sentiment Analysis", "left": "center"},
        "xAxis": {"type": "category", "data": ["Positive", "Negative"]},
        "yAxis": {"type": "value"},
        "series": [
            {
                "type": "bar",
                "data": [pos_score, neg_score],
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 1, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "#34e89e"},
                            {"offset": 1, "color": "#0f3443"}
                        ]
                    }
                }
            }
        ],
        "animationEasing": "bounceOut",
        "animationDelayUpdate": 300
    }

    st.markdown("### Sentiment Analysis")
    st_echarts(options=option, height="400px")


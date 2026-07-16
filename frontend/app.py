import streamlit as st
from PIL import Image

from api import predict_image

st.set_page_config(
    page_title="MedVision",
    layout="wide"
)

st.title("MedVision")

st.write("Chest X-Ray Disease Detection")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg","jpeg","png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    if uploaded_file:
        if st.button("Predict"):
            with st.spinner("Analyzing"):
                result = predict_image(uploaded_file)
                st.write(result)

            st.success(
                f"Prediction:{result['prediction']}"
            )
            st.metric(
                "confidence",
                f"{result['confidence']}%"
            )

            st.subheader("Class Probabilites")
            for cls,prob in result["probabilities"].items():
                st.progress(prob/100)
                st.write(f"{cls}: {prob}%")
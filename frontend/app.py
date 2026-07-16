import streamlit as st
from PIL import Image
from io import BytesIO
import base64
from api import predict_image

def base64_to_image(encoded):
    image_bytes = base64.b64decode(encoded)

    return Image.open(BytesIO(image_bytes))


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

    
    if st.button("Predict"):
            with st.spinner("Analyzing"):
                result = predict_image(uploaded_file)
                heatmap = base64_to_image(result["heatmap"])
                ##st.write(result)

            col1,col2 = st.columns(2)

            with col1:
                st.image(
                    image.resize((100,100)),
                    caption="Originail Uploaded X-Ray",
                    use_container_width=True
                )
            with col2:
                st.image(
                    heatmap.resize((100,100)),
                    caption="Heatmap(Grad-CAM)",
                    use_container_width=True
                )

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
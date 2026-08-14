import streamlit as st
from PIL import Image
from io import BytesIO
import base64

from api import predict_image, chat

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None



def base64_to_image(encoded):
    image_bytes = base64.b64decode(encoded)
    return Image.open(BytesIO(image_bytes))



st.set_page_config(
    page_title="MedVision",
    layout="wide"
)

st.title( "MedVision")
st.write("Chest X-Ray Disease Detection")


uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file:

    image = Image.open(uploaded_file)

    if st.button("Predict"):

        try:
            with st.spinner("Analyzing X-Ray..."):
                prediction_result = predict_image(uploaded_file)

            st.session_state.prediction_result = prediction_result
            st.session_state.uploaded_image = image
            st.session_state.messages = []

        except Exception as e:
            st.error(str(e))



if st.session_state.prediction_result is not None:

    result = st.session_state.prediction_result

    heatmap = base64_to_image(result["heatmap"])

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            st.session_state.uploaded_image.resize((100, 100)),
            caption="Original Uploaded X-Ray",
            width="stretch"
        )

    with col2:
        st.image(
            heatmap.resize((100,100)),
            caption="Grad-CAM Heatmap",
            width="stretch"
        )

    st.success(
        f"Prediction: {result['prediction']}"
    )

    st.metric(
        "Confidence",
        f"{result['confidence']}%"
    )

    st.subheader("Class Probabilities")

    for cls, prob in result["probabilities"].items():
        st.write(f"**{cls}**")
        st.progress(prob / 100)
        st.write(f"{prob}%")

    st.divider()

    st.subheader(" MedVision AI")



    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                with st.expander("Sources"):

                    for source in message["sources"]:

                        st.write(
                            f"**{source['id']}**  \n"
                            f"{source['source']} "
                            f"(Page {source['page']})"
                        )


    if prompt := st.chat_input("Ask MedVision AI..."):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("Thinking..."):

                chat_result = chat(
                    question=prompt,
                    prediction=result["prediction"]
                )

            answer = chat_result["answer"]

        except Exception as e:
            st.error(str(e))
            answer = None
        if answer:
            with st.chat_message("assistant"):

                st.markdown(answer)

                if chat_result["sources"]:

                    with st.expander("Sources"):

                        for source in chat_result["sources"]:

                            st.write(
                                f"**{source['id']}**  \n"
                                f"{source['source']} "
                                f"(Page {source['page']})"
                            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": chat_result["sources"]
                }
            )

else:

    st.info("Upload an X-ray and click **Predict** to start using MedVision AI.")
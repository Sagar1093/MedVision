import requests


API_URL = "http://127.0.0.1:8000/predict"
BASE_URL = "http://127.0.0.1:8000"

TIMEOUT = 60


def predict_image(image_file):
    files = {
        "file": (
            image_file.name,
            image_file.getvalue(),
            image_file.type
        )
    }

    try:
        response = requests.post(
            API_URL,
            files=files,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:
        raise Exception(
            "Unable to connect to the MedVision backend. "
            "Make sure FastAPI is running."
        )

    except requests.exceptions.Timeout:
        raise Exception(
            "The prediction request timed out. Please try again."
        )

    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get(
                "detail",
                "Backend error"
            )
        except Exception:
            detail = "Backend returned an error."

        raise Exception(
            f"Prediction failed: {detail}"
        )

    except requests.exceptions.RequestException as e:
        raise Exception(
            f"Prediction request failed: {e}"
        )


def chat(question, prediction):
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "question": question,
                "prediction": prediction
            },
            timeout=TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:
        raise Exception(
            "Unable to connect to the MedVision backend. "
            "Make sure FastAPI is running."
        )

    except requests.exceptions.Timeout:
        raise Exception(
            "The AI response timed out. Please try again."
        )

    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get(
                "detail",
                "Backend error"
            )
        except Exception:
            detail = "Backend returned an error."

        raise Exception(
            f"Chat request failed: {detail}"
        )

    except requests.exceptions.RequestException as e:
        raise Exception(
            f"Chat request failed: {e}"
        )


def stream_chat(question, prediction):

    response = requests.post(
        f"{BASE_URL}/chat/stream",
        json={
            "question": question,
            "prediction": prediction
        },
        stream=True,
        timeout=(10, 60)
    )

    response.raise_for_status()

    try:
        for line in response.iter_lines(
            decode_unicode=True
        ):
            if line:
                yield line

    finally:
        response.close()
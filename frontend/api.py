import requests

API_URL = "http://127.0.0.1:8000/predict"

def predict_image(image_file):
    files = {
        "file":(
            image_file.name,
            image_file.getvalue(),
            image_file.type
        )
    }

    response = requests.post(
        API_URL,
        files=files
    )
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)

    response.raise_for_status()

    return response.json()
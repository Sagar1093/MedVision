import requests

API_URL = "http://127.0.0.1:8000/predict"
BASE_URL = "http://127.0.0.1:8000"

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
    

    response.raise_for_status()

    return response.json()

def chat(question,prediction):
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "question":question,
            "prediction":prediction
        }
    )
    print("Question:", question)
    print("Prediction:", prediction)
    print(response.json())
    response.raise_for_status()

    return response.json()
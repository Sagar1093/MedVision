# MedVision AI

### Explainable AI for Chest X-Ray Classification

MedVision AI is a deep learning–based chest X-ray analysis system that classifies chest X-rays into three categories — **Normal**, **Pneumonia**, and **Tuberculosis** — and pairs each prediction with a visual explanation. It combines transfer learning, Grad-CAM interpretability, a FastAPI backend, a Streamlit frontend, and a Retrieval-Augmented Generation (RAG) chatbot into a single end-to-end research prototype.

> **Disclaimer:** MedVision AI is an academic and research project. It is **not clinically validated** and must not be used for medical diagnosis or treatment decisions.

---

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Model Comparison](#model-comparison)
7. [Training Configuration](#training-configuration)
8. [Model Evaluation](#model-evaluation)
9. [Explainable AI — Grad-CAM](#explainable-ai-with-grad-cam)
10. [Retrieval-Augmented Generation](#retrieval-augmented-generation)
11. [Conversational Interface](#conversational-interface)
12. [Technology Stack](#technology-stack)
13. [Results Summary](#results-summary)
14. [Project Structure](#project-structure)

---

## Features

| Category | Capability |
|---|---|
| Classification | Three-class chest X-ray classification (Normal / Pneumonia / Tuberculosis) |
| Model Selection | Benchmarked ResNet50, DenseNet121, and EfficientNet-B3; EfficientNet-B3 selected as final model |
| Explainability | Grad-CAM heatmaps with confidence scores and class probabilities |
| Backend | FastAPI inference API |
| Frontend | Streamlit web interface |
| Knowledge Assistant | RAG pipeline using Hugging Face BGE embeddings, ChromaDB, and MMR-based retrieval |
| Generation | Google Gemini for grounded, streaming responses with source citations |
| Performance | GPU-accelerated training with mixed precision |

---

## System Architecture

| Layer | Technology | Responsibilities |
|---|---|---|
| **Frontend** | Streamlit | Image upload, prediction display, confidence scores, class probabilities, Grad-CAM heatmaps, chat interface, source viewer |
| **Backend** | FastAPI | Image preprocessing, CNN inference, Grad-CAM generation, RAG request handling, response streaming |
| **Machine Learning** | EfficientNet-B3 | Final three-class chest X-ray classification |
| **RAG Pipeline** | LangChain, BGE embeddings, ChromaDB, MMR, Google Gemini | Retrieval and grounded response generation |

---

## Installation & Setup

### Prerequisites

| Requirement | Notes |
|---|---|
| Python | 3.9 or higher |
| pip | Latest version recommended |
| Git | For cloning the repository |
| CUDA-capable GPU | Optional, recommended for training / faster inference |
| Google Gemini API Key | Required for the RAG chatbot |

### 1. Clone the Repository

```bash
git clone https://github.com/Sagar1093/MedVision.git
cd MedVision
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate the environment
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Download / Place Trained Model Weights

Place the trained EfficientNet-B3 weights inside the `models/` directory. If you are training from scratch, see [Training Configuration](#training-configuration) for the parameters used in this project.

## Usage

### Start the Backend (FastAPI)

```bash
uvicorn backend.app:app --reload
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### Start the Frontend (Streamlit)

In a separate terminal:

```bash
streamlit run frontend/app.py
```

The app will be available at `http://localhost:8501`.

### Typical Workflow

| Step | Action |
|---:|---|
| 1 | Upload a chest X-ray image via the Streamlit interface |
| 2 | View the predicted class, confidence score, and class probabilities |
| 3 | Inspect the Grad-CAM heatmap overlay for model explainability |
| 4 | Ask the chatbot questions about the prediction or related conditions |
| 5 | Review the retrieved source documents behind each chatbot answer |

---

## Dataset

The project uses a chest X-ray dataset of **25,553 images** across three classes, split into training, validation, and test sets.

| Split | Normal | Pneumonia | Tuberculosis | Total |
|---|---:|---:|---:|---:|
| Train | 7,263 | 4,674 | 8,513 | **20,450** |
| Validation | 900 | 570 | 1,064 | **2,534** |
| Test | 925 | 580 | 1,064 | **2,569** |
| **Total** | **9,088** | **5,824** | **10,641** | **25,553** |

---

## Model Comparison

Three CNN architectures were trained under identical conditions using transfer learning with pretrained ImageNet weights.

| Rank | Model | Best Validation Accuracy |
|---:|---|---:|
| 1 | **EfficientNet-B3** | **77.86%** |
| 2 | DenseNet121 | 77.70% |
| 3 | ResNet50 | 77.66% |

**Final Model:** EfficientNet-B3 was selected based on its highest validation accuracy, though performance across all three architectures was closely matched.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Input Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 20 |
| Learning Rate | 1e-4 |
| LR Scheduler | ReduceLROnPlateau |
| LR Factor | 0.5 |
| LR Patience | 2 |
| Early Stopping Patience | 5 |
| Mixed Precision | Enabled |
| GPU | NVIDIA RTX 4060 Laptop GPU |

**Preprocessing & Augmentation**

| Stage | Transformations |
|---|---|
| Training | Resize (224×224), random horizontal flip, random rotation, tensor conversion, ImageNet normalization |
| Validation / Test | Resize (224×224), tensor conversion, ImageNet normalization (no augmentation) |

---

## Model Evaluation

Models were evaluated using accuracy, precision, recall, and F1 score, with validation accuracy as the primary comparison metric.

| Rank | Model | Validation Accuracy |
|---:|---|---:|
| 1 | EfficientNet-B3 | 77.86% |
| 2 | DenseNet121 | 77.70% |
| 3 | ResNet50 | 77.66% |

---

## Explainable AI with Grad-CAM

MedVision AI uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the image regions that most influenced each prediction. The resulting heatmap is displayed alongside the original X-ray in the Streamlit interface, giving users insight into the model's attention during classification.

> Grad-CAM is an interpretability technique — it does not establish clinical causality or confirm a medical diagnosis.

---

## Retrieval-Augmented Generation

MedVision AI includes a RAG pipeline that lets users ask questions about the supported diseases and the project itself, grounding Gemini's responses in a curated document collection rather than its internal knowledge alone.

### RAG Technology Stack

| Component | Technology |
|---|---|
| Document Processing | Python |
| Document Loading & Splitting | LangChain |
| Embeddings | BAAI/bge-small-en-v1.5 (Hugging Face) |
| Vector Database | ChromaDB |
| Retrieval Strategy | Maximal Marginal Relevance (MMR) |
| LLM | Google Gemini |
| Backend Integration | FastAPI |

### Retrieval Configuration

| Parameter | Value |
|---|---:|
| Retrieval Strategy | MMR |
| Retrieved Chunks | 4 |
| Candidate Chunks | 15 |
| Diversity Parameter | 0.7 |

### Pipeline Steps

| Step | Description |
|---:|---|
| 1 | Medical documents are collected and processed |
| 2 | Documents are split into smaller text chunks |
| 3 | Each chunk is embedded using `BAAI/bge-small-en-v1.5` |
| 4 | Embeddings are stored in ChromaDB |
| 5 | A user's question is converted into an embedding |
| 6 | ChromaDB retrieves candidate chunks via vector similarity |
| 7 | MMR selects four relevant, diverse chunks |
| 8 | Retrieved context is passed to Google Gemini |
| 9 | Gemini generates a grounded, streamed response |
| 10 | The application displays the answer alongside retrieved sources |

---

## Conversational Interface

The Streamlit chat interface supports questions such as:

- What are the symptoms of tuberculosis?
- What are the symptoms of pneumonia?
- What diagnostic methods are used?
- How does the model classify the X-ray?
- What does the Grad-CAM heatmap represent?

The chatbot receives the current CNN prediction as context, so questions about "the predicted condition" are interpreted correctly. Responses stream from Google Gemini, and retrieved source documents are displayed separately for inspection.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch, Torchvision |
| CNN Architectures | ResNet50, DenseNet121, EfficientNet-B3 |
| Explainability | Grad-CAM |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| RAG Framework | LangChain |
| Embeddings | Hugging Face, BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| LLM | Google Gemini |
| Data Processing | NumPy, Pandas, Scikit-learn |
| PDF Processing | PyPDF, PyMuPDF |
| GPU Acceleration | CUDA, PyTorch AMP |
| Version Control | Git, GitHub |

---

## Results Summary

| Metric | Value |
|---|---:|
| Total Images | 25,553 |
| Number of Classes | 3 |
| CNN Architectures Compared | 3 |
| Best Validation Accuracy | **77.86%** |
| Selected Model | **EfficientNet-B3** |

---

## Project Structure

```text
MedVision/
│
├── backend/
│   ├── app.py
│   ├── predictor.py
│   ├── schemas.py
│   ├── utils.py
│   │
│   └── rag/
│       ├── embeddings.py
│       ├── retriever.py
│       ├── rag_pipeline.py
│       └── db/
│
├── frontend/
│   ├── app.py
│   └── api.py
│
├── models/
├── datasets/
├── training/
├── evaluation/
│
├── requirements.txt
├── .gitignore
└── README.md
```
---

## Medical Disclaimer

MedVision AI is an academic and research prototype and has not been clinically validated.

The predictions generated by the classification model and the responses generated by the RAG-based assistant are not medical diagnoses and must not be used to make medical decisions.

Always consult a qualified healthcare professional for diagnosis, treatment, and medical advice.

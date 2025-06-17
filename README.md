# Arya: A Robust Digit Recognition & Analysis Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io)
[![Backend](https://img.shields.io/badge/Backend-PyTorch-orange.svg)](https://pytorch.org)

This project transcends the typical "Hello, World!" of machine learning—the MNIST classifier. It is a comprehensive exploration into engineering a production-conscious, fair, and resilient AI system. We move beyond simply achieving high accuracy to questioning the very nature of that accuracy: Is it robust? Is it equitable? Is the data it consumes reliable?

The result is **Arya**, an interactive web application that not only recognizes handwritten digits from an image but provides a transparent, analytical breakdown of its decision-making process.


## ► Core Features

*   **Multi-Digit Detection:** Intelligently segments and isolates multiple digits from a single uploaded image.
*   **High-Accuracy Recognition:** Leverages a robust, custom-trained Convolutional Neural Network (`Arya`) achieving >99% accuracy on the test set.
*   **Pre-inference Quality Assurance:** Automatically scans each detected digit for common quality issues like excessive rotation, noise, or blur *before* prediction.
*   **Transparent Analysis:** Provides a detailed, step-by-step report for each digit, including confidence scores, quality warnings, and pre-processing transformations.
*   **Interactive Controls:** Empowers the user with a "Retry" option for manual thresholding, offering a human-in-the-loop solution for challenging segmentations.

## ► The Philosophy: Beyond 99% Accuracy

A high accuracy metric is seductive but often misleading. A model's true value is revealed in its performance at the margins, with imperfect data, and across all strata of its domain. Our development philosophy was built on three pillars:

### 1. Model Sophistication.

The heart of the system is **`Arya`**, a bespoke Convolutional Neural Network engineered in PyTorch. The architecture wasn't chosen for complexity, but for stability and robustness.

*   **Architecture:** A thoughtful stack of `Conv2d` layers is paired with `BatchNorm2d`. Batch Normalization acts as a powerful regularizer, stabilizing the learning process and reducing sensitivity to initialization. This allows for faster, more reliable convergence.
*   **Efficiency:** An `AdaptiveAvgPool2d` layer is used before the final classifier. This provides flexibility for varying input sizes (though we standardize them) and creates a more efficient feature summary than a simple `Flatten` operation, contributing to a more generalized model.
*   **Training:** The model was trained not just to be accurate, but to be decisive, reflected in the high confidence scores it produces for clear inputs. The saved state `mnist_robust_cnn.pth` is the culmination of this rigorous training.

### 2. Algorithmic Fairness.

A model's global accuracy can mask significant local failures. Our offline `Fairness_Analysis.ipynb` toolkit was developed to dissect performance across discrete digit classes.

*   **The Why:** We asked: "Is our model equally good at recognizing a '4' as it is a '1'?" A model that is 99.9% accurate on '1's but only 92% accurate on '8's is not truly robust.
*   **The How:** By generating a per-class classification report (precision, recall, F1-score) and a confusion matrix, we verified that `Arya`'s performance is not just high, but also equitably distributed. This diligence ensures the model can be trusted across its entire problem space, preventing hidden biases.

### 3. Input Resilience.

Real-world data is messy. A model trained on pristine, centered MNIST digits will inevitably fail when faced with rotated, noisy, or blurry user-generated images. The `Quality_Checker.ipynb` module was built to address this "lab-to-live" gap.

*   **The Why:** Instead of letting the model silently fail on bad data, we built an intelligent gatekeeper. This system provides a better user experience and protects the integrity of the model's predictions.
*   **The How:** It's a rule-based expert system that runs before the model. It uses computer vision techniques (Laplacian variance for blur, Fourier analysis for noise estimation, contour properties for rotation) to flag potentially problematic inputs. This preemptive analysis transforms a potential model failure into a teachable moment for the user, complete with actionable feedback.

## ► System Architecture

The `webapp.py` application orchestrates these components into a seamless user experience.

`User Upload (Image)` -> `Image Pre-processing (Grayscale, Threshold)` -> `Contour Detection (Digit Segmentation)` -> **FOR EACH DIGIT:** [`Quality Check (Blur, Noise, Rotation)` -> `Normalization & Centering` -> `Arya Model Inference` -> `Confidence Calculation`] -> `Render Analysis Report`

---

## ► Getting Started

### Prerequisites

*   Python 3.11+
*   Git

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MachFrum/Week-3-A.I-For-S.E.git
    cd your_project_folder
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    The project relies on specific libraries. A `requirements.txt` file should be created with the necessary packages.
    ```bash
    # Create a requirements.txt file with:
    # streamlit
    # torch
    # torchvision
    # numpy
    # opencv-python-headless
    
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit Application:**
    ```bash
    streamlit run webapp.py
    ```

    Your browser should automatically open to the application's local URL.

## ► Project Structure

```
your_project_folder/
├── Arya.ipynb                # Notebook for model development, training, and evaluation.
├── Fairness_Analysis.ipynb     # Notebook for deep-diving into per-class model performance.
├── Quality_Checker.ipynb       # Notebook for developing the input image validation rules.
├── webapp.py                 # The final Streamlit application orchestrating the full pipeline.
├── README.md                 # This file.
├── Amazon Review Analysis    # There's a separate Read me here guiding you on the project scope.
├── Iris Analysis             # There's a separate Read me here guiding you on the project scope.
│
└── models/
    └── mnist_robust_cnn.pth    # The final, trained, and serialized PyTorch model state.
```

## ► Future Roadmap

*   **Real-time Canvas Input:** Allow users to draw digits directly in the browser.
*   **API Endpoint:** Expose the model via a REST API (using Flask or FastAPI) for programmatic access.
*   **Batch Processing:** Enable users to upload a zip file of images for bulk analysis.
*   **Deployment:** Containerize the application with Docker and deploy to a cloud service like Streamlit Community Cloud or Heroku.
# webapp.py (Final Version with Correct Indentation)

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os

class RobustCNN(nn.Module):
    def __init__(self):
        super(RobustCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

class ImageQualityValidator:
    def __init__(self):
        self.ROTATION_THRESHOLD = 20
        self.NOISE_THRESHOLD = 1500

    def _check_rotation(self, image):
        contours, _ = cv2.findContours((image > 0.1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        _, _, angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        if angle < -45:
            angle = 90 + angle

        if abs(angle) > self.ROTATION_THRESHOLD:
            return f"Digit may be rotated by ~{abs(angle):.0f} degrees."
        
        return None

    def _check_noise_and_blur(self, image):
        laplacian_var = cv2.Laplacian((image * 255).astype(np.uint8), cv2.CV_64F).var()
        if laplacian_var < 100:
            return f"Image may be blurry (var: {laplacian_var:.0f})."
        
        if laplacian_var > self.NOISE_THRESHOLD:
            return f"Image may be noisy (var: {laplacian_var:.0f})."
        
        return None

    def run(self, image_np):
        warnings = []
        if image_np.max() > 1.0:
            image_np = image_np / 255.0

        checks = [self._check_rotation, self._check_noise_and_blur]
        for check_func in checks:
            result = check_func(image_np)
            if result:
                warnings.append(result)
        
        return warnings

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustCNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}.")
        return None, None

def segment_and_preprocess_digits(image_pil, use_otsu, threshold_value):
    img_gray = np.array(image_pil.convert('L'))
    
    if use_otsu:
        _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, img_binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_digits = []
    for c in contours:
        if cv2.contourArea(c) > 25:
            x, y, w, h = cv2.boundingRect(c)
            if 0.1 < w/h < 2.0:
                detected_digits.append({'bbox': (x, y, w, h)})

    if not detected_digits:
        return []

    detected_digits.sort(key=lambda item: item['bbox'][0])
    processed_digits = []
    for digit in detected_digits:
        x, y, w, h = digit['bbox']
        img_digit_only = img_gray[y:y+h, x:x+w]
        canvas = np.zeros((max(w, h) + 20, max(w, h) + 20), dtype=np.uint8)
        start_x, start_y = (canvas.shape[1] - w) // 2, (canvas.shape[0] - h) // 2
        canvas[start_y:start_y+h, start_x:start_x+w] = img_digit_only
        final_image = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        img_tensor = transform(final_image).unsqueeze(0)
        processed_digits.append({'tensor': img_tensor, 'image': final_image})
    return processed_digits

def run_prediction(model, device, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    return prediction.item(), confidence.item()

def initialize_state():
    if 'initial_results' not in st.session_state:
        st.session_state.initial_results = None
    if 'retry_results' not in st.session_state:
        st.session_state.retry_results = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

def reset_session():
    st.session_state.initial_results = None
    st.session_state.retry_results = None
    st.session_state.uploaded_file = None

def display_results(title, results_list, container):
    if not results_list:
        return
    final_str = "".join([str(res['prediction']) for res in results_list])
    container.header(title)
    container.success(f"Detected Digits: **{final_str}**")
    container.code(final_str, language="text")
    tab_titles = [f"Digit #{i+1} (Pred: {res['prediction']})" for i, res in enumerate(results_list)]
    tabs = container.tabs(tab_titles)
    for i, res in enumerate(results_list):
        with tabs[i]:
            st.metric(f"Prediction", value=f"{res['prediction']}", delta=f"{res['confidence']:.2%} confidence")
            st.image(res['image'], caption=f"Processed image for Digit #{i+1}")
            st.subheader("Quality Validation")
            if not res['warnings']:
                st.success("✅ No quality issues detected.")
            else:
                for warning in res['warnings']:
                    st.warning(warning)

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Interactive Digit Recognizer")
initialize_state()

with st.sidebar:
    st.header("About This App")
    st.write("This app demonstrates an interactive multi-digit recognition pipeline.")
    st.subheader("Components:")
    st.markdown("- **Digit Segmentation:** Uses OpenCV to find digits.")
    st.markdown("- **`RobustCNN` (`Arya`):** Classifies each segmented digit.")
    st.markdown("- **`ImageQualityValidator` (`Jaqen`):** Checks each digit for quality issues.")
    st.divider()
    st.write("Built by GROUP 29.")

st.title("🤖 Interactive Multi-Digit Recognizer")
st.markdown("Upload an image with handwritten digits. You can then 'Retry' with different settings to improve the result.")

MODEL_PATH = "models/mnist_robust_cnn.pth"
model, device = load_model(MODEL_PATH)
validator = ImageQualityValidator()

col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
with col2:
    st.write("") 
    st.button("Clear All & Reset", on_click=reset_session, use_container_width=True)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file:
    image = Image.open(st.session_state.uploaded_file)
    st.image(image, caption="Current Uploaded Image", width=400)
    
    if st.session_state.initial_results is None:
        with st.spinner("Running initial analysis..."):
            segmented_digits = segment_and_preprocess_digits(image, use_otsu=True, threshold_value=128)
            if segmented_digits:
                results = []
                for digit_data in segmented_digits:
                    pred, conf = run_prediction(model, device, digit_data['tensor'])
                    warnings = validator.run(digit_data['image'])
                    results.append({'prediction': pred, 'confidence': conf, 'warnings': warnings, 'image': digit_data['image']})
                st.session_state.initial_results = results

    if st.session_state.initial_results:
        display_results("Initial Pass (Automatic Settings)", st.session_state.initial_results, st)

        st.divider()
        st.header("⚙️ Retry with Manual Settings")
        st.info("If the initial result is incorrect, adjust the threshold to change how the model separates digits from the background, then click 'Retry'.")
        
        retry_threshold = st.slider("Segmentation Threshold", 0, 255, 128)
        
        if st.button("Retry Analysis"):
            with st.spinner("Re-analyzing with manual settings..."):
                segmented_digits = segment_and_preprocess_digits(image, use_otsu=False, threshold_value=retry_threshold)
                if segmented_digits:
                    results = []
                    for digit_data in segmented_digits:
                        pred, conf = run_prediction(model, device, digit_data['tensor'])
                        warnings = validator.run(digit_data['image'])
                        results.append({'prediction': pred, 'confidence': conf, 'warnings': warnings, 'image': digit_data['image']})
                    st.session_state.retry_results = results
                else:
                    st.session_state.retry_results = []

    if st.session_state.retry_results is not None:
        if not st.session_state.retry_results:
            st.warning("No digits were found with the current manual threshold setting.")
        else:
            display_results("Retry Pass (Manual Settings)", st.session_state.retry_results, st)
else:
    st.info("Upload an image to begin.")
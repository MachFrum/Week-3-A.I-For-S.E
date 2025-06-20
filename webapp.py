import streamlit as st
import torch
import numpy as np
import cv2
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# --- Define the model class (should match your notebook definition) ---
import torch.nn as nn
class RobustCNN(nn.Module):
    def __init__(self):
        super(RobustCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
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

@st.cache_resource
def load_model(model_path):
    device = torch.device("cpu")
    model = RobustCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- Helper functions ---

def segment_digits(image):
    # image: numpy array (H, W, 3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 5)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > 100:
            roi = morph[y:y+h, x:x+w]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            digits.append(roi)
            bboxes.append((x, y, w, h))
    # Sort left to right
    bboxes, digits = zip(*sorted(zip(bboxes, digits), key=lambda t: t[0][0])) if digits else ([],[])
    return list(digits), list(bboxes)

def classify_digit_rois(digits, model):
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    model.eval()
    results = []
    for roi in digits:
        tensor = transform(roi).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            pred = out.argmax(dim=1).item()
            confidence = torch.softmax(out, 1)[0, pred].item()
        results.append((pred, confidence))
    return results

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

def analyze_and_visualize(all_targets, all_preds):
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)               
    plot_confusion_matrix(cm)
    report = classification_report(all_targets, all_preds, output_dict=True, digits=3)
    st.write('### Detailed Classification Report (Latest)')
    df = pd.DataFrame(report).transpose()
    st.dataframe(df)

# --- Streamlit UI: App Layout ---

st.set_page_config(page_title="Multi-digit MNIST OCR", layout="centered")
st.title("🖼️ Multi-Digit Handwritten Number Recognition")
st.markdown(
    """
    Upload your photo of handwritten or printed digits. 
    The system will extract, classify, and display all detected numbers.
    """
)

MODEL_PATH = "models/mnist_robust_cnn.pth"
if not os.path.exists(MODEL_PATH):
    st.error(f"Could not find model at {MODEL_PATH}. Please train and save your model first.")
    st.stop()

with st.status("Initialising Model...", expanded=True) as status:
    model = load_model(MODEL_PATH)
    st.write("Model loaded and ready.")
    status.update(label="Model loaded.", state="complete")

# --- Optionally show last model analytics (run this block once offline and paste results here) ---
if "analytics" not in st.session_state:
    # Load last test analytics if available (adapt to your storage method)
    # If you want to skip this live, comment this block.
    try:
        import pickle
        with open("mnist_test_preds_targets.pkl", "rb") as f:
            preds, targets = pickle.load(f)
        st.session_state.analytics = (targets, preds)
    except Exception:
        st.session_state.analytics = None

if st.session_state.analytics:
    st.header("📊 Model Test Performance")
    analyze_and_visualize(*st.session_state.analytics)
    st.markdown("---")

# ---------------- Upload & Inference --------------------
uploaded = st.file_uploader("Upload a digit image (jpg, png, etc.)", type=['jpg','jpeg','png'])
if uploaded:
    image_bytes = uploaded.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    cv_img = np.array(pil_image)[..., ::-1]
    st.image(pil_image, caption="Uploaded image", use_container_width=True)
    
    phases = ["Preprocessing", "Digit Segmentation", "Digit Classification", "Generating Result"]
    
    with st.status("Processing...", expanded=True) as phase_status:
        # Everything in this block (lines below) must be indented!

        st.write(":hourglass: Preprocessing image ...")
        phase_status.update(label=f"{phases[0]} done.", state="running")
        
        digits, bboxes = segment_digits(cv_img)
        st.write(f":mag_right: Found {len(digits)} digit-like regions.")
        phase_status.update(label=f"{phases[1]} done.", state="running")
        if len(digits) == 0:
            st.warning("No digits found in this image. Try a clearer/cropped sample.")
            st.stop()

        results = classify_digit_rois(digits, model)
        st.write(f":pencil: Recognized {len(results)} digits.")
        phase_status.update(label=f"{phases[2]} done.", state="running")

        predicted_str = ''.join(str(result[0]) for result in results)
        st.write(f":clipboard: Recognized number sequence: **{predicted_str}**")
        phase_status.update(label=f"{phases[3]}", state="complete")

    # Now, OUTSIDE the 'with' block, dedent
    draw_img = cv_img.copy()
    for (digit, conf), (x, y, w, h) in zip(results, bboxes):
        cv2.rectangle(draw_img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(draw_img, str(digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,20,60), 2)
    st.image(draw_img[..., ::-1], caption="Detected digits", use_container_width=True)

    st.markdown("### 📋 Copy Recognized Numbers")
    st.code(predicted_str, language=None)

    import pandas as pd
    digits_table = pd.DataFrame(
        [(str(d), f"{conf:.2f}") for d, conf in results],
        columns=["Digit", "Confidence"]
    )
    st.dataframe(digits_table, use_container_width=True)

st.markdown("---")
st.markdown("Made by Peter Macharia. Powered by PyTorch & Streamlit.")
# MNIST Digit Classification with CNN

## 📌 Goal

Build a **Convolutional Neural Network (CNN)** to classify handwritten digits from the MNIST dataset.

### Objectives:

* Achieve **>95% test accuracy**
* Visualize the model’s predictions on 5 sample images
* Deliver a complete solution including:

  * Model architecture
  * Training and evaluation loops
  * Visualization and performance analysis

---

## 🧠 Approach

This project uses **PyTorch** to implement and train a CNN for digit classification on the MNIST dataset.

### 🔍 Steps Overview:

1. **Data Preparation**

   * MNIST dataset loaded via `torchvision.datasets`
   * Input images normalized using dataset-specific mean and standard deviation for stable training

2. **Model Architecture**

   * A simple yet effective CNN with:

     * Two convolutional layers
     * Two fully connected layers
     * ReLU activations and max pooling
     * Dropout for regularization

3. **Training Loop**

   * Optimizer: **Adam**
   * Loss Function: **CrossEntropyLoss**
   * Metrics: Accuracy and loss tracked over epochs
   * Trained for **5 epochs**

4. **Evaluation**

   * Calculated average loss and accuracy on test set
   * Achieved **>99% test accuracy**, surpassing the original target

5. **Visualization**

   * Plotted training and test loss/accuracy across epochs
   * Visualized 5 sample predictions with color-coded correctness
   * Generated confusion matrix and detailed classification report

---

## 📈 Results

| Metric               | Value                                   |
| -------------------- | --------------------------------------- |
| Final Test Accuracy  | **99.28%**                              |
| Final Train Accuracy | 99.20%                                  |
| Epochs Trained       | 5                                       |
| Visualization        | ✅ Sample predictions + confusion matrix |

---

## 📊 Sample Visual Output

* **Training Curves**: Shows model convergence over time
* **Sample Predictions**: Displays predicted vs. true labels for 5 test images
* **Confusion Matrix**: Confirms balanced performance across all digits
* **Classification Report**: Precision, recall, and F1-score per class

---

## 🔧 How to Run

```bash
# 1. Clone this repository
git clone https://github.com/MachFrum/Week-3-A.I-For-S.E.git
cd mnist-cnn-pytorch

# 2. Install required packages
pip install torch torchvision matplotlib seaborn scikit-learn

# 3. Run the script
python mnist_cnn.py
```

---

## 🧾 File Structure

```
mnist_cnn.py             # Full training pipeline
README.md                # Project documentation
```

---

## ✅ Key Takeaways

* Simple CNNs can achieve state-of-the-art performance on structured image data like MNIST
* Proper regularization and visualization help validate and trust model behavior
* Modular code design ensures reusability and readability

---

## 📬 Contact

For any questions or suggestions, feel free to reach out via GitHub or email.


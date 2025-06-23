# 🌸 Iris Species Classification with Decision Tree

A comprehensive machine learning project that uses Decision Tree Classifier to classify iris flowers into three species: Setosa, Versicolor, and Virginica. This project demonstrates the complete machine learning pipeline from data exploration to model evaluation and visualization.

## 📊 Project Overview

This project performs classification analysis on the famous Iris dataset using a Decision Tree Classifier. The analysis includes exploratory data analysis, model training, performance evaluation, and beautiful visualizations to understand the decision-making process.

## 🎯 Key Results

- **Perfect Accuracy**: Achieved 100% accuracy on the test set
- **Balanced Dataset**: 50 samples per species (150 total)
- **Clean Data**: No missing values or preprocessing required
- **Interpretable Model**: Decision tree provides clear decision rules

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **seaborn** & **matplotlib** - Data visualization
- **scikit-learn** - Machine learning algorithms and tools
- **numpy** - Numerical computing

## 📁 Project Structure

```
├── iris_classification.py    # Main analysis script
├── README.md                # Project documentation
```

## 🔍 Code Breakdown

### 1. Data Loading and Setup

```python
# Loading the famous Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Converting to DataFrame for easier manipulation
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['species'] = iris.target_names[y]
```

### 2. Exploratory Data Analysis (EDA)

The analysis begins with comprehensive data exploration:

#### Dataset Overview
- **Features**: 4 numerical features (sepal length, sepal width, petal length, petal width)
- **Target**: 3 iris species (setosa, versicolor, virginica)
- **Samples**: 150 total (50 per species)

#### Key Findings
| Species | Count | Distribution |
|---------|-------|--------------|
| Setosa | 50 | 33.3% |
| Versicolor | 50 | 33.3% |
| Virginica | 50 | 33.3% |

#### Visualization Components
- **Pairplot**: Shows relationships between all feature pairs
- **Boxplots**: Displays feature distributions across species
- **Feature Importance**: Identifies most discriminative features

### 3. Data Preprocessing

```python
# Clean dataset - no missing values detected
# Data already numerically encoded
# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Model Training

```python
# Initialize Decision Tree with reproducible results
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)
```

### 5. Model Evaluation

#### Performance Metrics

```
🎯 Accuracy: 100%

📊 Detailed Classification Report:
              precision    recall  f1-score   support
    setosa       1.00      1.00      1.00        10
versicolor       1.00      1.00      1.00         9
 virginica       1.00      1.00      1.00        11

  accuracy                           1.00        30
 macro avg       1.00      1.00      1.00        30
weighted avg     1.00      1.00      1.00        30
```

#### Key Performance Indicators
- ✅ **Perfect Precision**: 100% for all species
- ✅ **Perfect Recall**: 100% for all species
- ✅ **Perfect F1-Score**: 100% for all species
- ✅ **No False Positives or Negatives**

### 6. Feature Importance Analysis

The decision tree analysis reveals which features are most important for classification:

```python
# Extract and rank feature importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)
```

### 7. Decision Tree Visualization

Two levels of visualization are provided:
- **Complete Tree**: Shows all decision nodes and splits
- **Top 3 Levels**: Focuses on the most important decision points

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/MachFrum/Week-3-A.I-For-S.E.git
   cd iris-classification
   ```

2. **Install dependencies**
   ```bash
   pip install pandas seaborn matplotlib scikit-learn numpy
   ```

3. **Run the analysis**
   ```bash
   python iris_classification.py
   ```

## 📈 Visualizations Generated

The script generates several informative visualizations:

1. **Pairplot** - Shows feature relationships and species separation
2. **Boxplots** - Displays feature distributions by species
3. **Feature Importance Bar Chart** - Ranks features by importance
4. **Decision Tree Diagrams** - Visualizes the decision-making process

## 🧠 Key Insights

- **Perfect Classification**: The decision tree achieved 100% accuracy, indicating clear separability between iris species
- **Feature Discrimination**: Some features are more discriminative than others for species classification
- **Balanced Dataset**: Equal representation of all species ensures unbiased model training
- **Interpretability**: Decision tree provides clear, human-readable classification rules

## 🔬 Model Characteristics

- **Algorithm**: Decision Tree Classifier
- **Criterion**: Gini impurity (default)
- **Max Depth**: Unlimited (allows full tree growth)
- **Random State**: 42 (ensures reproducible results)
- **Test Size**: 20% of total dataset

## 🎓 Learning Outcomes

This project demonstrates:
- Complete machine learning pipeline implementation
- Effective data visualization techniques
- Model evaluation and interpretation
- Feature importance analysis
- Decision tree visualization and understanding

## 📚 Dataset Information

The Iris dataset is a classic dataset in machine learning, containing:
- 150 samples of iris flowers
- 4 features: sepal length, sepal width, petal length, petal width
- 3 target classes: Iris setosa, Iris versicolor, Iris virginica
- Originally collected by Edgar Anderson and made famous by Ronald Fisher

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements. Suggestions for additional analysis or visualization techniques are welcome!

## 📄 License

This project is open source and available for anyone.

---

*Created with ❤️ for machine learning education and demonstration by group 29*

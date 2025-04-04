
# Bank Marketing Prediction Project

![enter image description here](https://www.pngplay.com/wp-content/uploads/2/Bank-PNG-Photos.png)

## 📌 Overview
Machine learning model to predict if bank clients will subscribe to term deposits based on marketing campaign data. This Flask web application serves predictions via a user-friendly interface.

## 📂 Dataset Information

### Sources
- **Creators**: Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL) @ 2012
- **Original Study**: [Using Data Mining for Bank Direct Marketing](https://www.researchgate.net/publication/228346283_Using_Data_Mining_for_Bank_Direct_Marketing)

### Key Details
- **Instances**: 45,211 (full dataset)
- **Attributes**: 16 input variables + 1 output (y)
- **Prediction Target**: Will client subscribe to term deposit? (binary: "yes"/"no")

### Attributes Description
| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | age | numeric | Client's age |
| 2 | job | categorical | Type of job (admin, management, etc.) |
| 3 | marital | categorical | Marital status (married, divorced, etc.)|
| 4 | education  | categorical | Education Level (secondary, primary, tertiary, etc.) |
| 5 | default  | binary | has credit in default?|
| 6 | balance  | numeric | average yearly balance, in euros |
| 7 | housing  | binary | has housing loan? |
| 8 | loan | binary | has personal loan? |
| 9 | contact | categorical | contact communication type (unknown, telephone, etc.) |
| 10 | day | numeric | last contact day of the month |
| 11 | month | categorical | last contact month of year (jan, feb, mar, etc.)|
| 12 | duration | numeric | last contact duration in seconds |
| 13 | campaign | numeric | number of contacts performed during this campaign and for this client |
| 14 | pdays | numeric | number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted) |
| 15 | previous | numeric | number of contacts performed before this campaign and for this clients |
| 16 | poutcome | categorical | outcome of the previous marketing campaign (failure, success, etc.)|
| ... | ... | ... | ... |
| 17 | y | binary | Target: subscribed to deposit? |

## 📦 Project Structure

bank-marketing/
├── models/               # Pretrained models
│   ├── model1.pkl
│   └── model2.pkl
├── templates/            # HTML templates
├── app.py                # Flask application
├── setup.config          # Configuration
└── requirements.txt      # Dependencies


## 📊 Results

### Prediction Goal
The classification goal is to predict if the client will subscribe to a term deposit (variable `y`).

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Use Case |
|-------|----------|-----------|--------|----------|----------|
| **Model 1: Hypertuned CatBoost** | 0.9029 | 0.5622 | 0.7668 | 0.6487 | Maximize subscriber identification |
| **Model 2: Calibrated CatBoost** | 0.9143 | 0.6558 | 0.5633 | 0.6060 | Balanced approach |
| **Model 3: Threshold-Adjusted (0.6)** | 0.9113 | 0.6943 | 0.4321 | 0.5327 | High-confidence predictions |

#### Model 1: Initial Hypertuned CatBoost Classifier
**High Recall Focus**  
- Designed to capture maximum potential subscribers (high recall)
- Tolerates some false positives
- **Best when**: The priority is identifying every possible subscriber, even with some incorrect classifications

#### Model 2: Calibrated CatBoost Classifier  
**Balanced Precision/Recall**  
- Optimizes both precision and recall
- Fewer false positives than Model 1
- **Best when**: You need reliable predictions without extreme bias toward recall or precision

#### Model 3: Threshold-Adjusted (0.6) CatBoost  
**High Precision Focus**  
- Only predicts "subscribe" when ≥60% confidence
- Minimizes false positives
- **Best when**: Accuracy of positive predictions is critical, even if some subscribers are missed


## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- pip

### Quick Start
```bash
git clone https://github.com/yourusername/bank-marketing-prediction.git
cd bank-marketing-prediction
pip install -r requirements.txt
python app.py
```

## 📜 License
MIT License https://choosealicense.com/licenses/mit/


## 🧚🏼‍♂️ Donation --> Way to heaven
If you appreciate this project and want to support future work, consider buying me [☕](https://buymeacoffee.com/prasadpandp)... (or better, donating a [GPU](https://www.amazon.in/gp/cart/view.html?ref_=nav_cart) 😆).

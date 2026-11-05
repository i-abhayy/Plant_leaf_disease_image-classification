# 🌿 LeafCare AI
### Plant Leaf Disease Detection and Care Recommendation System

LeafCare AI is an AI-powered plant leaf disease detection system that uses **MobileNetV2 Deep Learning** and **Google Gemini AI** to analyze plant leaf images, identify possible diseases, and provide care recommendations.

The project combines image classification with AI-assisted visual assessment to help users understand potential plant health issues.

---

## ✨ Features

- **38-Class Disease Classification:** Identifies plant diseases across 14 plant types using a trained MobileNetV2 model.
- **Image Upload:** Upload a plant leaf image for analysis.
- **AI-Powered Assessment:** Uses Google Gemini to inspect the uploaded image and provide an additional assessment.
- **Disease Explanation:** Displays information about the predicted condition.
- **Possible Causes:** Provides potential reasons behind the observed symptoms.
- **Care Recommendations:** Suggests practical steps to help care for affected plants.
- **Prevention Tips:** Provides suggestions to reduce the risk of future disease.
- **Prediction Confidence:** Displays the MobileNetV2 model's confidence score.
- **Responsive Interface:** Built using HTML, CSS, and JavaScript.

## 🧠 Machine Learning Model

The project uses **MobileNetV2**, a convolutional neural network architecture based on transfer learning.

| Model Information | Details |
|---|---|
| Architecture | MobileNetV2 |
| Framework | TensorFlow / Keras |
| Image Input Size | 224 × 224 |
| Number of Classes | 38 |
| Plant Types | 14 |
| Training Dataset | PlantVillage |
| Model File | `multiplant_mobilenetv2_best.keras` |

### Model Evaluation

The model was evaluated on a held-out test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 96.61% |
| Macro Precision | 95.97% |
| Macro Recall | 95.30% |
| Macro F1-Score | 95.56% |
| Weighted F1-Score | 96.58% |

*These results represent the model's evaluation on the test dataset. Real-world performance may vary depending on image quality, lighting, plant varieties, and environmental conditions.*

## 🛠️ Technologies Used

**Frontend**
- HTML5
- CSS3
- JavaScript

**Backend**
- Python
- Flask

**Machine Learning**
- TensorFlow
- Keras
- MobileNetV2
- NumPy

**Generative AI**
- Google Gemini API

**Development Tools**
- VS Code
- Jupyter Notebook
- Git and GitHub

## 📂 Project Structure

```text
LeafCare-AI/
│
├── app.py
├── .gitignore
│
├── models/
│   ├── class_names.json
│   └── multiplant_mobilenetv2_best.keras
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cnn_model.ipynb
│   └── 03_mobilenetv2_model.ipynb
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── templates/
    └── index.html
```

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/i-abhayy/Plant_leaf_disease_image-classification.git
```

### 2. Navigate to the Project Folder

```bash
cd Plant_leaf_disease_image-classification
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install flask tensorflow numpy pillow google-genai python-dotenv
```

Use Python and TensorFlow versions compatible with your environment.

### 5. Configure Gemini API

Create a `.env` file in the root project directory:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-flash-latest
```

Replace `your_gemini_api_key` with your own API key.

Never upload your actual `.env` file or API key to GitHub.

You can obtain a Gemini API key through [Google AI Studio](https://aistudio.google.com/apikey).

### 6. Run the Application

```bash
python app.py
```

Open the local URL displayed in your terminal, typically:

```text
https://plant-leaf-disease-image-classification.vercel.app/
```

## 📸 Screenshots

Add screenshots of your application here.

You can create a `screenshots/` folder and add your interface images.

For example:

```markdown
![LeafCare AI Homepage](screenshots/homepage.png)
![Disease Detection Results](screenshots/results.png)
```

## ⚠️ Important Notes

- The trained model supports 38 predefined plant disease and healthy-leaf classes.
- Plants or diseases outside the training classes may not be identified correctly.
- A high model confidence score does not guarantee a correct diagnosis.
- Gemini's assessment is an additional AI-generated opinion, not a medical or agricultural laboratory diagnosis.
- Always inspect the plant and consult an agricultural expert when necessary.
- The original training dataset and temporary training files are not included in this repository.

## 🔮 Future Improvements

- Improve detection on real-world plant images.
- Expand the supported plant and disease categories.
- Add more detailed disease-specific care guidance.
- Improve deployment and application performance.
- Add additional plant health analysis capabilities.

## 👨‍💻 Developer

**Abhay Yadav**

- GitHub: [@i-abhayy](https://github.com/i-abhayy)
- LinkedIn: [Abhay Yadav](https://www.linkedin.com/in/i-abhay/)

---

⭐ If you find this project interesting, consider giving the repository a star!

**LeafCare AI | Helping plants stay healthy through AI and Deep Learning.** 🌱

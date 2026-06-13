# 🛍️ E-Commerce Product Review Sentiment Analyzer

An end-to-end Machine Learning pipeline and Graphical User Interface (GUI) that classifies e-commerce customer reviews into **Positive**, **Neutral**, or **Negative** sentiments. 

This project demonstrates a complete Natural Language Processing (NLP) workflow, from raw text cleaning and dataset balancing to model training and interactive deployment.

## ✨ Key Features
* **Custom NLP Preprocessing:** Implements text stemming, regex cleaning, and a customized stopword filter that preserves negation words (e.g., "didn't", "not") to maintain sentiment context.
* **Dataset Balancing:** Addresses class imbalance (a common issue in review datasets) using **Undersampling**, ensuring the model learns fair representations of all sentiment classes rather than just guessing the majority class.
* **3-Way Classification:** Maps a standard 1-5 star rating system into a robust 3-class target variable (Negative, Neutral, Positive).
* **Interactive Desktop GUI:** Built with Tkinter, allowing users to type a custom review and receive an instant sentiment prediction and color-coded UI response.

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Natural Language Processing:** `nltk` (PorterStemmer, Stopwords), `re` (Regular Expressions)
* **Machine Learning:** `scikit-learn` (Random Forest Classifier, CountVectorizer, train_test_split)
* **Model Serialization:** `joblib`
* **User Interface:** `tkinter`

## 📊 Dataset & Modeling Strategy
* **Data Source:** E-Commerce Product Reviews (e.g., Flipkart/Amazon).
* **Feature Engineering:** Combined `summary` and `review` text columns to maximize the vocabulary context.
* **Vectorization:** `CountVectorizer` limited to the top 5,000 most frequent words to optimize memory and training speed.
* **Algorithm:** Random Forest Classifier (`n_estimators=100`).

## 🚀 Installation & Setup

 **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/ecommerce-sentiment-analyzer.git](https://github.com/yourusername/ecommerce-sentiment-analyzer.git)
   cd ecommerce-sentiment-analyzer
```


**Install required dependencies:**

pip install pandas numpy scikit-learn nltk joblib
Download NLTK Stopwords:
import nltk
nltk.download('stopwords')

**Prepare the Data & Train the Model:**

Place your dataset (e.g., Product_Reviews.csv) in the root directory.
Run the Jupyter Notebook (ECommerce_Product_Review_Sentiment_Analyzer.ipynb) sequentially to clean the data, balance the dataset, train the model, and export the cv.pkl and model.pkl files.

💻 Usage
Once the .pkl files are generated, you can launch the interactive GUI:

python your_gui_script_name.py
(Replace your_gui_script_name.py with the name of the python file containing your Tkinter code).

A desktop window will open.
Type a product review into the text box.
Click "Analyze Sentiment".
The prediction will instantly appear at the bottom of the window (🟢 Positive, 🟡 Neutral, or 🔴 Negative).

**📈 Performance Metrics**

Testing Split: 20% unseen data.
Evaluation: Evaluated using standard classification metrics including Accuracy, Precision, Recall, and a Weighted F1-Score to accurately reflect multi-class performance.

**🔮 Future Improvements**

Deploy the saved models to the web using Flask or FastAPI instead of a local Tkinter app.
Experiment with TF-IDF Vectorization instead of standard Count Vectorization.
Introduce deep learning techniques (e.g., LSTMs or pre-trained Transformers like BERT) for higher contextual understanding.
  

# Fashion Recommendation Prediction Pipeline

## Project Overview

StyleSense is an online women's clothing retailer that receives thousands of product reviews from customers. While customers frequently provide detailed written feedback, many reviews do not include the **"Recommended" indicator**, which tells whether the customer recommends the product.

The goal of this project is to build a **machine learning pipeline** that predicts whether a customer recommends a product based on:

* Review text
* Customer demographic information
* Product category information

The pipeline combines **Natural Language Processing (NLP)** techniques with structured data preprocessing to automatically infer the recommendation label from the available information.

This solution demonstrates how a **scalable ML pipeline** can integrate multiple data types (text, categorical, and numerical) into a single model workflow.

---

## Dataset

The dataset contains **18,442 customer reviews** from a women's clothing e-commerce platform.

### Features

| Feature                 | Description                                                                      |
| ----------------------- | -------------------------------------------------------------------------------- |
| Clothing ID             | Unique identifier for the product                                                |
| Age                     | Age of the reviewer                                                              |
| Title                   | Review title                                                                     |
| Review Text             | Full written review                                                              |
| Positive Feedback Count | Number of users who found the review helpful                                     |
| Division Name           | High-level product category                                                      |
| Department Name         | Product department                                                               |
| Class Name              | Product class                                                                    |
| Recommended IND         | Target variable indicating recommendation (1 = recommended, 0 = not recommended) |

For the modeling process:

* **Title** and **Review Text** were combined into a single feature called `combined_review`.
* `Clothing ID` was removed because it does not provide predictive information.

---

## Machine Learning Pipeline

This project implements a **scikit-learn pipeline** to ensure that all preprocessing steps and the model training occur in a single reproducible workflow.

### Pipeline Architecture

```
Raw Data
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
ColumnTransformer
   ├ Numeric Features → StandardScaler
   ├ Categorical Features → OneHotEncoder
   └ Text Features → spaCy preprocessing → TF-IDF
   ↓
Logistic Regression Classifier
   ↓
GridSearchCV Hyperparameter Tuning
   ↓
Model Evaluation
```

---

## NLP Processing

Text data is processed using **spaCy** to perform linguistic preprocessing before vectorization.

The following NLP steps are applied:

* Tokenization
* Lemmatization
* Stopword removal
* Punctuation removal

After preprocessing, **TF-IDF vectorization** converts the cleaned text into numerical features suitable for machine learning models.

---

## Model

The final model used in the pipeline is **Logistic Regression**.

Reasons for choosing Logistic Regression:

* Performs well on high-dimensional sparse text features
* Efficient and interpretable baseline for NLP classification tasks
* Works effectively with TF-IDF representations

To address class imbalance in the dataset, the classifier uses:

```
class_weight = "balanced"
```

This ensures that the model pays additional attention to the minority class (non-recommended reviews).

---

## Hyperparameter Tuning

Model performance was improved using **GridSearchCV** to tune the regularization parameter of Logistic Regression.

Example parameter grid:

```
classifier__C = [0.1, 1, 10]
```

Cross-validation (`cv = 3`) was used to evaluate model configurations and select the best performing model.

---

## Model Performance

Final evaluation on the test set produced the following results:

| Metric    | Class 0 (Not Recommended) | Class 1 (Recommended) |
| --------- | ------------------------- | --------------------- |
| Precision | 0.48                      | 0.95                  |
| Recall    | 0.80                      | 0.82                  |
| F1-Score  | 0.60                      | 0.88                  |

Overall model accuracy:

```
Accuracy ≈ 81%
```

### Interpretation

* The model performs very well at identifying **recommended products**.
* Detection of **non-recommended products** significantly improved after applying class balancing.
* The pipeline provides a strong baseline for recommendation prediction from customer reviews.

---

## Technologies Used

* Python
* pandas
* numpy
* scikit-learn
* spaCy
* Jupyter Notebook

---

## Project Structure

```
dsnd-pipelines-project/
│
├── starter/
│   └── reviews.csv
│
├── project_notebook.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/your-username/dsnd-pipelines-project.git
```

2. Navigate to the project directory:

```
cd dsnd-pipelines-project
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Install the spaCy language model:

```
python -m spacy download en_core_web_sm
```

5. Run the Jupyter notebook:

```
jupyter notebook
```

---

## Key Learnings

This project demonstrates:

* Building **scikit-learn pipelines for mixed data types**
* Integrating **NLP preprocessing with machine learning models**
* Handling **class imbalance in classification tasks**
* Performing **hyperparameter tuning using GridSearchCV**
* Evaluating models using **precision, recall, and F1 score**

The final pipeline provides a clean and scalable architecture for predicting product recommendations from customer review data.

---

## Future Improvements

Possible improvements include:

* Using **more advanced NLP features** (POS tags, sentiment scores)
* Testing **additional classifiers** such as SVM or Gradient Boosting
* Implementing **deep learning models** (e.g., BERT-based text classifiers)
* Building an interactive **dashboard for review prediction**

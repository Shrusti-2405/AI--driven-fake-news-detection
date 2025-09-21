import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(file_path)
    df = df.dropna()
    df['text'] = df['title'] + " " + df['text']
    df['label'] = df['label'].map({0: 'fake', 1: 'real'})
    return df

# --- Main part of the script ---

# 1. Load data
df = load_data('WELFake_Dataset.csv')

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 3. Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# 4. Transform the text data into TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 5. Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 6. Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save the model and the vectorizer
joblib.dump(model, 'fake_news_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

print("\nModel and vectorizer saved!")
import joblib

# Load the saved model and vectorizer
model = joblib.load('fake_news_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def classify_article(article_text):
    """Classifies a given article as 'real' or 'fake'."""
    # Transform the input text using the loaded vectorizer
    article_tfidf = vectorizer.transform([article_text])

    # Make a prediction
    prediction = model.predict(article_tfidf)

    return prediction[0]

# --- Main interactive loop ---
if __name__ == "__main__":
    print("Fake News Detector")
    print("Enter a news article to classify. Type 'exit' to quit.")

    while True:
        # Take input from the user
        user_input = input("\n> ")

        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Classify the input and print the result
        classification = classify_article(user_input)
        print(f"  -> The article is classified as: {classification.upper()}")
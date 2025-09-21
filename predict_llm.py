from transformers import pipeline

# Load the zero-shot-classification pipeline
# This will download the model the first time you run it
print("Loading model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("Model loaded.")

def classify_article_llm(article_text):
    """Classifies an article using a zero-shot LLM."""
    labels = ["real", "fake"]
    result = classifier(article_text, candidate_labels=labels)

    # The result is a dictionary containing scores
    # We return the label with the highest score
    return result['labels'][0]

# --- Main interactive loop ---
if __name__ == "__main__":
    print("\nFake News Detector (LLM Version)")
    print("Enter a news article to classify. Type 'exit' to quit.")

    while True:
        user_input = input("\n> ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Check for very short input
        if len(user_input.split()) < 5:
            print("  -> Please enter a more complete sentence or article for better results.")
            continue

        classification = classify_article_llm(user_input)
        print(f"  -> The article is classified as: {classification.upper()}")
import os
from flask import Flask, request, render_template, jsonify
from transformers import pipeline
from serpapi import GoogleSearch
import time # To simulate loading for testing

# Initialize the Flask application
app = Flask(__name__)

# --- API Key for Search ---
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    print("Warning: SERPAPI_API_KEY environment variable not set. Search functionality will be disabled.")

# --- Load Final Local LLM ---
print("Loading local language model... This is a one-time process when the server starts.")
try:
    fact_checker_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    print("Model loaded successfully. The server is ready.")
except Exception as e:
    print(f"Failed to load model. Error: {e}")
    # We can let the app run without the model for UI development if needed
    fact_checker_pipeline = None

def get_search_evidence(claim):
    if not SERPAPI_API_KEY:
        return "Search is disabled. Please set the SERPAPI_API_KEY."
    try:
        params = {"q": claim, "api_key": SERPAPI_API_KEY, "num": 5}
        search = GoogleSearch(params)
        results = search.get_dict()
        evidence = [f"Source: {res.get('title')}\nSnippet: {res.get('snippet')}" for res in results.get("organic_results", [])]
        return "\n\n".join(evidence)
    except Exception as e:
        print(f"Error during search: {e}")
        return "Could not retrieve search results."

def fact_check_claim_local(claim, evidence):
    if not fact_checker_pipeline:
        return "Model not loaded. Cannot perform fact-checking."
    
    prompt = f"""
    Question: Based on the context provided, is the following claim true, false, or misleading?
    Claim: "{claim}"
    Context:
    ---
    {evidence}
    ---
    Answer:
    """
    
    try:
        response = fact_checker_pipeline(prompt, max_length=512)
        analysis_text = response[0]['generated_text']
        return analysis_text
    except Exception as e:
        print(f"Error during local LLM call: {e}")
        return "Error: Could not generate a response from the local model."

# --- Page Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fact-checker')
def fact_checker_page():
    return render_template('fact_checker.html')

@app.route('/about')
def about():
    return render_template('about.html')

# --- API Endpoint for Fact-Checking ---
@app.route('/api/fact-check', methods=['POST'])
def api_fact_check():
    data = request.get_json()
    claim = data.get('claim', '')
    
    if not claim or len(claim) < 5:
        return jsonify({'error': 'Please provide a more substantial claim.'}), 400

    evidence = get_search_evidence(claim)
    if "Could not retrieve" in evidence or "Search is disabled" in evidence:
        return jsonify({'error': evidence}), 500

    result = fact_check_claim_local(claim, evidence)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

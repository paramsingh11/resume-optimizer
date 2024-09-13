from flask import Flask, request, jsonify, send_from_directory
import spacy
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return set(tokens)

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings

def compute_semantic_similarity(text1, text2):
    emb1 = get_bert_embeddings(text1)
    emb2 = get_bert_embeddings(text2)
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
    return similarity.item()

def compute_ats_friendly_score(resume, job_description):
    job_keywords = preprocess_text(job_description)
    resume_keywords = preprocess_text(resume)

    matched_keywords = len(job_keywords & resume_keywords)
    total_keywords = len(job_keywords)
    ats_score = (matched_keywords / total_keywords) * 100 if total_keywords > 0 else 0

    semantic_similarity = compute_semantic_similarity(resume, job_description)

    return ats_score, f"ATS-Friendly score: {ats_score:.2f}%\nSemantic Similarity: {semantic_similarity:.2f}"

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        data = request.json
        resume_text = data.get('resume', '')
        job_description = data.get('job_description', '')

        if not resume_text or not job_description:
            return jsonify({'error': 'Both resume and job description must be provided.'}), 400

        resume_words = preprocess_text(resume_text)
        job_words = preprocess_text(job_description)

        overlap = len(resume_words & job_words)
        total = len(job_words)
        match_score = (overlap / total) * 100 if total > 0 else 0
        match_feedback = f"Overlap score: {match_score:.2f}%"
        
        ats_score, ats_feedback = compute_ats_friendly_score(resume_text, job_description)

        response = {
            'match_score': match_score,
            'match_feedback': match_feedback,
            'ats_score': ats_score,
            'ats_feedback': ats_feedback,
            'semantic_similarity': compute_semantic_similarity(resume_text, job_description)
        }

        return jsonify(response)
    except Exception as e:
        print(f"Error: {e}")  # Debugging line
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)

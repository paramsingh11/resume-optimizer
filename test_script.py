from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def calculate_ats_score(resume, job_description):
    # Keyword matching using CountVectorizer
    vectorizer = CountVectorizer().fit_transform([resume, job_description])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    match_score = cosine_sim[0, 1] * 100  # Convert to percentage

    # Semantic similarity using spaCy
    resume_doc = nlp(resume)
    job_desc_doc = nlp(job_description)
    semantic_similarity = resume_doc.similarity(job_desc_doc)

    # Simple ATS Score calculation
    ats_score = (match_score * 0.5 + semantic_similarity * 100 * 0.5)  # Weighted average

    return ats_score, match_score, semantic_similarity

# Example data
resume_text = """ Enter Your Resume
"""

job_description_text = """ Enter Your Job Description
"""

# Calculate the scores
ats_score, match_score, semantic_similarity = calculate_ats_score(resume_text, job_description_text)

print(f"ATS Score: {ats_score:.2f}%")
print(f"Match Score: {match_score:.2f}%")
print(f"Semantic Similarity: {semantic_similarity:.2f}")

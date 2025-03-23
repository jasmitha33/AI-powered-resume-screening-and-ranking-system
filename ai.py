import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


nlp = spacy.load("en_core_web_sm")


resumes = [
    "Experienced software engineer with proficiency in Python, JavaScript, and machine learning.",
    "Data scientist with a strong background in machine learning, data analysis, and deep learning.",
    "Project manager with expertise in managing teams, business analysis, and delivering projects on time.",
    "Senior software developer skilled in Java, Python, Agile methodologies, and problem-solving."
]


job_description = """
We are looking for a software engineer proficient in Python, JavaScript, and machine learning.
Experience with deep learning is a plus. Strong problem-solving skills and ability to work in a team are essential.
"""


def preprocess_text(text):
    doc = nlp(text.lower())  
    tokens = [token.text for token in doc if token.text not in stopwords.words('english') and token.is_alpha]
    return " ".join(tokens)


processed_resumes = [preprocess_text(resume) for resume in resumes]
processed_job_description = preprocess_text(job_description)


vectorizer = TfidfVectorizer()


corpus = [processed_job_description] + processed_resumes


tfidf_matrix = vectorizer.fit_transform(corpus)


cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()


resume_scores = list(zip(resumes, cosine_similarities))


sorted_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)


print("Ranked Resumes:")
for i, (resume, score) in enumerate(sorted_resumes):
    print(f"Rank {i+1}:")
    print(f"Resume: {resume}")
    print(f"Score: {score:.2f}")
    print("-" * 50)


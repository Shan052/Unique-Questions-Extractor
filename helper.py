import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Load trained SBERT model ONCE
# ------------------------------
MODEL_PATH = r"C:\Users\DELL\PycharmProjects\question_similarity\trained_sbert\sbert_minilm_local"
# OR (better option if helper.py is in same folder):
# MODEL_PATH = "trained_duplicate_bert"

model = SentenceTransformer(MODEL_PATH)

# ------------------------------
# Extract questions from PDF
# ------------------------------
def extract_questions_from_pdf(pdf_file):
    questions = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                line = line.strip()
                if len(line) > 10:
                    questions.append(line)

    return questions

# ------------------------------
# Create embedding
# ------------------------------
def get_embedding(text):
    return model.encode(text, convert_to_numpy=True)

# ------------------------------
# Duplicate check
# ------------------------------
def is_duplicate(q1, q2, threshold=0.80):
    emb1 = get_embedding(q1).reshape(1, -1)
    emb2 = get_embedding(q2).reshape(1, -1)

    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity >= threshold, similarity

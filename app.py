import streamlit as st
import helper

st.set_page_config(page_title="Duplicate Question Detector", layout="wide")

st.title("📄 PDF Unique Question Extractor ")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

threshold = 0.80

if uploaded_file:
    with st.spinner("Extracting questions from PDF..."):
        questions = helper.extract_questions_from_pdf(uploaded_file)

    st.success(f"Total questions found: {len(questions)}")

    unique_questions = []
    duplicates = []

    with st.spinner("Detecting duplicates using BERT..."):
        for q in questions:
            duplicate_found = False

            for uq in unique_questions:
                is_dup, score = helper.is_duplicate(q, uq, threshold)

                if is_dup:
                    duplicates.append((q, uq, score))
                    duplicate_found = True
                    break

            if not duplicate_found:
                unique_questions.append(q)

    st.success(f"Unique questions found: {len(unique_questions)}")

    # ------------------------------
    # Display Unique Questions
    # ------------------------------
    st.subheader("✅ Unique Questions")
    for i, q in enumerate(unique_questions, 1):
        st.write(f"{i}. {q}")

    # ------------------------------
    # Optional: Show duplicates
    # ------------------------------
    with st.expander("🔁 Show Detected Duplicates"):
        for d in duplicates:
            st.write(f"❌ `{d[0]}`")
            st.write(f"↳ Similar to: `{d[1]}`")
            st.write(f"Similarity Score: **{d[2]:.2f}**")
            st.divider()

    # ------------------------------
    # Download
    # ------------------------------
    st.download_button(
        "⬇ Download Unique Questions",
        data="\n".join(unique_questions),
        file_name="unique_questions.txt",
        mime="text/plain"
    )

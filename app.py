import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE SETUP ----------------

st.set_page_config(
    page_title="Dense Embedding-Based Semantic Search",
    page_icon="🔎",
    layout="wide"
)

# ---------------- LOAD CUSTOM CSS ----------------

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )
    
# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ---------------- DOCUMENT COLLECTION ----------------

documents = [
    "Artificial intelligence enables computers to perform tasks that normally require human intelligence.",
    "Machine learning allows computers to learn patterns from data and make predictions.",
    "Deep learning uses neural networks with multiple layers to learn complex patterns.",
    "Natural language processing enables computers to understand and process human language.",
    "Semantic search retrieves information based on meaning rather than exact keyword matches.",
    "Dense embeddings represent text as numerical vectors that capture semantic meaning.",
    "Cosine similarity measures how similar two vectors are based on their orientation.",
    "Information retrieval systems help users find relevant documents from large collections.",
    "Database systems store and organize information so that it can be efficiently retrieved.",
    "Computer vision enables machines to analyze and understand images and visual information."
]


# ---------------- SESSION STATE ----------------

if "document_embeddings" not in st.session_state:
    st.session_state.document_embeddings = None

if "search_results" not in st.session_state:
    st.session_state.search_results = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ---------------- KEYWORD SEARCH ----------------

def keyword_search(query, documents, top_k):

    query_words = set(
        re.findall(r"\b\w+\b", query.lower())
    )

    scores = []

    for document in documents:

        document_words = set(
            re.findall(r"\b\w+\b", document.lower())
        )

        matching_words = query_words.intersection(
            document_words
        )

        scores.append(len(matching_words))

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    return ranked_indices[:top_k], scores


# ---------------- TITLE ----------------

st.title("Dense Embedding-Based Semantic Search")

st.write("Virtual Laboratory Experiment")


# ---------------- SIDEBAR ----------------

st.sidebar.title("Experiment Sections")

section = st.sidebar.radio(
    "Select a section:",
    [
        "Purpose",
        "Theory",
        "Simulation",
        "Quiz",
        "Report Generation",
        "Certificate",
        "References"
    ]
)


st.header(section)


# ---------------- PURPOSE ----------------

if section == "Purpose":

    st.subheader("Aim")

    st.write(
        "To develop a semantic search system using dense vector embeddings "
        "to retrieve documents that are conceptually similar to a user's query."
    )

    st.subheader("Objectives")

    st.markdown("""
    - Generate dense vector embeddings for documents and queries.
    - Measure similarity between the query and documents.
    - Retrieve the most semantically relevant documents.
    - Compare semantic search with traditional keyword-based search.
    """)


# ---------------- THEORY ----------------

if section == "Theory":

    st.subheader("1. What is Semantic Search?")

    st.write(
        "Semantic search retrieves documents based on their meaning rather than "
        "only matching the exact words present in the query."
    )

    st.subheader("2. Dense Embeddings")

    st.write(
        "A dense embedding represents a document or query as a numerical vector. "
        "Texts with similar meanings tend to have similar vector representations."
    )

    st.subheader("3. Similarity Measurement")

    st.write(
        "The similarity between the query vector and document vectors is measured "
        "using cosine similarity. A higher similarity score indicates greater "
        "semantic similarity."
    )

    st.subheader("4. Retrieval")

    st.write(
        "The system calculates similarity scores for the documents and ranks them. "
        "The top-k documents with the highest scores are returned as the search results."
    )

    st.subheader("Basic Workflow")

    st.markdown("""
    **Documents → Dense Embeddings → Vector Representation**

    **User Query → Dense Embedding → Similarity Calculation → Ranking → Top-k Results**
    """)

    st.subheader("Key Terms")

    terms = {
        "Dense Embedding":
            "A numerical vector representing the meaning of text.",

        "Semantic Search":
            "Search based on meaning and context.",

        "Cosine Similarity":
            "A measure of similarity between two vectors.",

        "Top-k Retrieval":
            "Selecting the k documents with the highest similarity scores."
    }

    for term, definition in terms.items():

        st.markdown(
            f"**{term}:** {definition}"
        )


# ---------------- SIMULATION ----------------

if section == "Simulation":

    st.subheader("Semantic Search Simulation")


    # STEP 1
    st.subheader("Step 1: Document Collection")

    st.write(
        "The following documents are used for the semantic search experiment:"
    )

    for i, document in enumerate(documents, start=1):

        st.write(
            f"**D{i}:** {document}"
        )


    # STEP 2
    st.subheader("Step 2: Generate Dense Embeddings")

    if st.button("Generate Document Embeddings"):

        st.session_state.document_embeddings = model.encode(
            documents
        )

        st.success(
            "Dense embeddings generated successfully!"
        )

        st.write(
            f"Each document has been converted into a "
            f"{st.session_state.document_embeddings.shape[1]}-dimensional vector."
        )


    # STEP 3
    st.subheader("Step 3: Enter Search Query")

    query = st.text_input(
        "Enter your search query:",
        placeholder="Example: How do computers understand human language?"
    )

    top_k = st.slider(
        "Number of results to retrieve:",
        min_value=1,
        max_value=5,
        value=3
    )


    # STEP 4
    st.subheader("Step 4: Retrieve Similar Documents")

    if st.button("Search"):

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        elif st.session_state.document_embeddings is None:

            st.warning(
                "Please generate the document embeddings "
                "before performing a search."
            )

        else:

            # Generate query embedding
            query_embedding = model.encode(
                [query]
            )


            # Calculate cosine similarity
            similarity_scores = cosine_similarity(
                query_embedding,
                st.session_state.document_embeddings
            )[0]


            # Rank documents
            ranked_indices = similarity_scores.argsort()[::-1]


            # Select top-k documents
            top_indices = ranked_indices[:top_k]


            st.success(
                "Semantic search completed!"
            )


            # ---------------- SEMANTIC RESULTS ----------------

            st.subheader("Search Results")

            results = []

            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                results.append({
                    "Rank": rank,
                    "Document": f"D{index + 1}",
                    "Similarity Score": round(
                        float(similarity_scores[index]),
                        4
                    )
                })


            results_df = pd.DataFrame(
                results
            )
            st.session_state.search_results = results_df
            st.session_state.last_query = query

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )


            # ---------------- BAR CHART ----------------

            fig = go.Figure(
                go.Bar(
                    x=results_df["Document"],
                    y=results_df["Similarity Score"],
                    text=results_df["Similarity Score"],
                    textposition="auto"
                )
            )


            fig.update_layout(
                title="Semantic Similarity Scores",
                xaxis_title="Document",
                yaxis_title="Cosine Similarity",
                yaxis_range=[0, 1]
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            # ---------------- KEYWORD SEARCH ----------------

            st.subheader(
                "Keyword Search Comparison"
            )


            keyword_indices, keyword_scores = keyword_search(
                query,
                documents,
                top_k
            )


            for rank, index in enumerate(
                keyword_indices,
                start=1
            ):

                st.write(
                    f"**{rank}. D{index + 1}** - "
                    f"{documents[index]}"
                )

                st.write(
                    f"Keyword Matches: "
                    f"{keyword_scores[index]}"
                )

                st.divider()


            # ---------------- COMPARISON TABLE ----------------

            st.subheader(
                "Semantic vs Keyword Search"
            )


            comparison = []


            for rank in range(top_k):

                semantic_index = top_indices[rank]

                keyword_index = keyword_indices[rank]


                comparison.append({

                    "Rank":
                        rank + 1,

                    "Semantic Search":
                        f"D{semantic_index + 1}",

                    "Semantic Score":
                        round(
                            float(
                                similarity_scores[
                                    semantic_index
                                ]
                            ),
                            4
                        ),

                    "Keyword Search":
                        f"D{keyword_index + 1}",

                    "Keyword Matches":
                        keyword_scores[
                            keyword_index
                        ]
                })


            comparison_df = pd.DataFrame(
                comparison
            )


            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )
            st.subheader("Observation")

            best_semantic_index = top_indices[0]

            best_keyword_index = keyword_indices[0]

            st.info(
                f"Semantic search ranked D{best_semantic_index + 1} as the "
                f"most relevant document with a cosine similarity score of "
                f"{similarity_scores[best_semantic_index]:.4f}."
            )

            if best_semantic_index != best_keyword_index:

                st.write(
                    "The top result differs between semantic search and keyword "
                    "search. This shows that semantic search can identify "
                    "conceptually related documents even when the exact query "
                    "words are not present."
                )

            else:

                st.write(
                    "Both methods returned the same top document for this query. "
                    "However, semantic search ranks documents using their "
                    "meaning represented by dense embeddings."
                )


# ---------------- QUIZ ----------------

if section == "Quiz":

    st.subheader("Semantic Search Quiz")

    quiz_questions = [
        {
            "question": "What is the main purpose of semantic search?",
            "options": [
                "To match only exact keywords",
                "To retrieve documents based on meaning",
                "To store documents in a database",
                "To convert documents into images"
            ],
            "answer": "To retrieve documents based on meaning"
        },
        {
            "question": "What is a dense embedding?",
            "options": [
                "A database table",
                "A numerical vector representing text",
                "A keyword list",
                "A text file"
            ],
            "answer": "A numerical vector representing text"
        },
        {
            "question": "Which model is used in this experiment to generate embeddings?",
            "options": [
                "all-MiniLM-L6-v2",
                "ResNet-50",
                "BERT-Base-Cased",
                "GPT-2"
            ],
            "answer": "all-MiniLM-L6-v2"
        },
        {
            "question": "What does cosine similarity measure?",
            "options": [
                "The length of a document",
                "The number of words in a query",
                "The similarity between two vectors",
                "The number of documents"
            ],
            "answer": "The similarity between two vectors"
        },
        {
            "question": "What does a higher cosine similarity score generally indicate?",
            "options": [
                "Greater semantic similarity",
                "More spelling errors",
                "A longer document",
                "Fewer keywords"
            ],
            "answer": "Greater semantic similarity"
        },
        {
            "question": "What is the purpose of Top-k retrieval?",
            "options": [
                "To retrieve the k most relevant documents",
                "To remove all documents",
                "To generate k embeddings",
                "To count keywords"
            ],
            "answer": "To retrieve the k most relevant documents"
        },
        {
            "question": "What is the main difference between keyword search and semantic search?",
            "options": [
                "Keyword search uses images",
                "Semantic search uses exact words only",
                "Keyword search focuses on word matching while semantic search considers meaning",
                "There is no difference"
            ],
            "answer": "Keyword search focuses on word matching while semantic search considers meaning"
        },
        {
            "question": "What is generated for the user's search query?",
            "options": [
                "A database",
                "A dense embedding",
                "A PDF file",
                "A new document collection"
            ],
            "answer": "A dense embedding"
        },
        {
            "question": "How are documents ranked in semantic search?",
            "options": [
                "By document length",
                "By alphabetical order",
                "By similarity score",
                "By upload time"
            ],
            "answer": "By similarity score"
        },
        {
            "question": "Why can semantic search find conceptually related documents?",
            "options": [
                "Because embeddings capture semantic information",
                "Because every document contains the same words",
                "Because documents are sorted alphabetically",
                "Because it ignores the query"
            ],
            "answer": "Because embeddings capture semantic information"
        }
    ]


    # ---------------- QUESTIONS ----------------

    for i, question in enumerate(quiz_questions, start=1):

        with st.container(border=True):

            st.markdown(
                f'<div class="quiz-question-label">Question {i:02d}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="quiz-question-text">{question["question"]}</div>',
                unsafe_allow_html=True
            )

            st.radio(
                "Select your answer:",
                question["options"],
                key=f"quiz_{i}"
            )


    # ---------------- SUBMIT ----------------

    if st.button("Submit Quiz"):

        score = 0

        for i, question in enumerate(quiz_questions, start=1):

            selected_answer = st.session_state.get(
                f"quiz_{i}"
            )

            if selected_answer == question["answer"]:

                score += 1


        st.session_state.quiz_score = score

        st.success(
            f"Quiz submitted! Your score is {score}/10.",
            icon=None
        )


        # ---------------- FEEDBACK ----------------

        st.subheader("Quiz Feedback")

        for i, question in enumerate(quiz_questions, start=1):

            selected_answer = st.session_state.get(
                f"quiz_{i}"
            )

            if selected_answer == question["answer"]:

                st.markdown(
                    f"""
                    <div class="feedback-row feedback-correct">
                        <span class="feedback-number">{i:02d}</span>
                        <span class="feedback-question">Question {i}</span>
                        <span class="feedback-status">Correct</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="feedback-row feedback-review">
                        <span class="feedback-number">{i:02d}</span>
                        <span class="feedback-question">Question {i}</span>
                        <span class="feedback-status">Review</span>
                    </div>
                    <div class="feedback-answer">
                        Correct answer: <strong>{question['answer']}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ---------------- REPORT GENERATION ----------------

if section == "Report Generation":

    st.subheader("Experiment Report")

    student_name = st.text_input(
        "Student Name"
    )

    roll_number = st.text_input(
        "Roll Number"
    )

    st.write("### Experiment Details")

    st.write(
        "**Experiment:** Dense Embedding-Based Semantic Search"
    )

    st.write(
        "**Aim:** To develop a semantic search system using dense vector "
        "embeddings to retrieve conceptually similar documents."
    )

    st.subheader("Search Details")

    if st.session_state.search_results is not None:

        st.write(
            f"**Search Query:** {st.session_state.last_query}"
        )

        st.write(
            f"**Documents in Collection:** {len(documents)}"
        )

        st.write(
            f"**Results Retrieved:** "
            f"{len(st.session_state.search_results)}"
        )

        st.write("### Retrieved Documents")

        st.dataframe(
            st.session_state.search_results,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Complete a semantic search in the Simulation section "
            "to include search results in the report."
        )
    st.subheader("Quiz Performance")

    if "quiz_score" in st.session_state:

        st.write(
            f"**Quiz Score:** "
            f"{st.session_state.quiz_score}/10"
        )

    else:

        st.info(
            "Complete the quiz to include the quiz score in the report."
        )

    st.subheader("Generate Report")

    if st.button("Generate PDF Report"):

        if not student_name.strip() or not roll_number.strip():

            st.warning(
                "Please enter your name and roll number first."
            )

        elif st.session_state.search_results is None:

            st.warning(
                "Please complete the semantic search before generating the report."
            )

        else:

            from fpdf import FPDF
            from datetime import datetime

            pdf = FPDF()
            pdf.add_page()

            pdf.set_font("Arial", "B", 16)
            pdf.cell(
                0,
                10,
                "Virtual Laboratory Experiment Report",
                ln=True,
                align="C"
            )

            pdf.ln(8)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(
                0,
                8,
                "Dense Embedding-Based Semantic Search",
                ln=True
            )

            pdf.ln(5)

            pdf.set_font("Arial", "", 11)

            pdf.cell(
                0,
                8,
                f"Student Name: {student_name}",
                ln=True
            )

            pdf.cell(
                0,
                8,
                f"Roll Number: {roll_number}",
                ln=True
            )

            pdf.cell(
                0,
                8,
                f"Date: {datetime.now().strftime('%d-%m-%Y')}",
                ln=True
            )

            pdf.ln(5)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(
                0,
                8,
                "Experiment Details",
                ln=True
            )

            pdf.set_font("Arial", "", 11)

            pdf.multi_cell(
                0,
                7,
                "Aim: To develop a semantic search system using "
                "dense vector embeddings to retrieve conceptually "
                "similar documents."
            )

            pdf.ln(4)

            pdf.cell(
                0,
                8,
                f"Search Query: {st.session_state.last_query}",
                ln=True
            )

            pdf.cell(
                0,
                8,
                f"Documents in Collection: {len(documents)}",
                ln=True
            )

            pdf.ln(5)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(
                0,
                8,
                "Retrieved Documents",
                ln=True
            )

            pdf.set_font("Arial", "", 11)

            for _, row in st.session_state.search_results.iterrows():

                pdf.multi_cell(
                    180,
                    7,
                    f"Rank {row['Rank']} - "
                    f"{row['Document']} - "
                    f"Similarity Score: {row['Similarity Score']}"
                )

                pdf.ln(2)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(
                0,
                8,
                "Quiz Performance",
                ln=True
            )

            pdf.set_font("Arial", "", 11)

            quiz_score = st.session_state.get(
                "quiz_score",
                "Not completed"
            )

            pdf.cell(
                0,
                8,
                f"Quiz Score: {quiz_score}/10",
                ln=True
            )

            pdf.ln(5)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(
                0,
                8,
                "Conclusion",
                ln=True
            )

            pdf.set_font("Arial", "", 11)

            pdf.multi_cell(
                0,
                7,
                "The experiment demonstrates how dense embeddings "
                "can be used to represent text numerically and retrieve "
                "documents based on semantic similarity."
            )

            pdf_bytes = bytes(pdf.output())

            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="semantic_search_report.pdf",
                mime="application/pdf"
            )

# ---------------- CERTIFICATE ----------------

if section == "Certificate":

    st.subheader("Virtual Laboratory Certificate")

    certificate_name = st.text_input(
        "Enter your name for the certificate:"
    )

    if st.button("Generate Certificate"):

        if not certificate_name.strip():

            st.warning(
                "Please enter your name first."
            )

        else:

            from fpdf import FPDF

            pdf = FPDF(
                orientation="L",
                unit="mm",
                format="A4"
            )
            pdf.set_margins(18, 16, 18)
            pdf.add_page()

            page_width = 297
            page_height = 210

            pdf.set_draw_color(35, 55, 70)
            pdf.set_line_width(1.2)
            pdf.rect(10, 10, page_width - 20, page_height - 20)

            pdf.set_draw_color(173, 123, 53)
            pdf.set_line_width(0.5)
            pdf.rect(15, 15, page_width - 30, page_height - 30)

            pdf.set_text_color(35, 55, 70)
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 12, "VIRTUAL LABORATORY", align="C", ln=True)

            pdf.set_text_color(173, 123, 53)
            pdf.set_font("Times", "B", 28)
            pdf.cell(0, 18, "CERTIFICATE OF COMPLETION", align="C", ln=True)

            pdf.set_draw_color(173, 123, 53)
            pdf.line(106, 58, 191, 58)
            pdf.ln(10)

            pdf.set_text_color(78, 85, 90)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 9, "This certificate is proudly presented to", align="C", ln=True)
            pdf.ln(3)

            pdf.set_text_color(35, 55, 70)
            pdf.set_font("Times", "B", 27)
            pdf.cell(0, 18, certificate_name.strip(), align="C", ln=True)

            pdf.set_draw_color(173, 123, 53)
            pdf.line(82, 100, 215, 100)
            pdf.ln(9)

            pdf.set_text_color(78, 85, 90)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, "for successfully completing the virtual laboratory experiment", align="C", ln=True)

            pdf.set_text_color(35, 55, 70)
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 12, "Dense Embedding-Based Semantic Search", align="C", ln=True)

            pdf.set_text_color(78, 85, 90)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(
                0,
                7,
                "The learner demonstrated an understanding of dense embeddings, "
                "cosine similarity, semantic search, and top-k retrieval.",
                align="C"
            )

            pdf.set_y(166)
            pdf.set_draw_color(78, 85, 90)
            pdf.line(48, 178, 105, 178)
            pdf.line(192, 178, 249, 178)
            pdf.set_font("Arial", "", 9)
            pdf.set_text_color(78, 85, 90)
            pdf.set_xy(48, 180)
            pdf.cell(57, 6, "Virtual Laboratory", align="C")
            pdf.set_xy(192, 180)
            pdf.cell(57, 6, "Instructor", align="C")

            certificate_pdf = bytes(pdf.output())

            st.success(
                f"Certificate generated successfully for {certificate_name}!",
                icon=None
            )

            st.markdown(
                f"""
                ### Certificate of Completion

                This is to certify that

                ## **{certificate_name}**

                has successfully completed the Virtual Laboratory experiment

                **Dense Embedding-Based Semantic Search**

                and demonstrated an understanding of dense embeddings,
                cosine similarity, semantic search, and top-k retrieval.
                """
            )

            st.download_button(
                label="Download Certificate PDF",
                data=certificate_pdf,
                file_name="virtual_lab_certificate.pdf",
                mime="application/pdf"
            )


# ---------------- REFERENCES ----------------

if section == "References":

    st.subheader("References")

    references = [
        (
            "IIT Kharagpur Virtual Lab",
            "https://vlab.co.in/"
        ),
        (
            "Sentence Transformers Documentation",
            "https://sbert.net/"
        ),
        (
            "Scikit-learn Documentation",
            "https://scikit-learn.org/"
        ),
        (
            "Streamlit Documentation",
            "https://docs.streamlit.io/"
        ),
        (
            "FPDF2 Documentation",
            "https://py-pdf.github.io/fpdf2/"
        )
    ]

    for i, (name, link) in enumerate(references, start=1):

        st.markdown(
            f"{i}. **{name}**  \n"
            f"[{link}]({link})"
        )
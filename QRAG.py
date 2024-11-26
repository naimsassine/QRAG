import pandas as pd
import numpy as np
import re
import io
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from google.colab import files

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


ATTACHMENT_TYPE_ERR_MSG = (
    "All attachments must be either DctmObjRef or Attachment type, got {}: {}"
)
ATTACH_TYPE_EXPECTED = "Attachment expected to be of type `Attachment`, got {}"
UNEXPECTED_ATTR_TO_PARSE = (
    "Attribute to parse from attachments expected to be in "
    "['body', 'filename'], got '{}'"
)
DCTM_OBJ_REF_EXPECTED = "Expected DctmObjRef, got {}: {}"
DOXC2TXT_EXCEPTION = "Cannot process file, raised '{}' error"
LIST_OR_STR_ATTACH_EXPECTED = "Got type {} for attachment, only list or str accepted"
PAGE_SEP = "\n" + "=" * 31 + " NEW PAGE " + "=" * 31 + "\n"
MISSING_SPACES_PATTERNS = [
    "IndicativeTermsheet\n",
    "PRIVATEPLACEMENT\n",
    "PublicOfferingonlyin:",
]


def check_txt_missing_spaces(all_pages_txt: str, threshold: float = 0.06) -> bool:
    """Check if the parsed PDF has missing spaces (as for all Leonteq termsheets).

    Notes
    -----
    The alignment used to format the Leonteq termsheets are not properly recognized by
    our PDF converter. As an undesirable result, most spaces are being removed during
    the conversion step leading to erroneous extractions.
    """

    nb_spaces = all_pages_txt.count(" ")
    nb_chars = len(all_pages_txt)
    ratio = nb_spaces / nb_chars

    return ratio < threshold and any(
        p in all_pages_txt for p in MISSING_SPACES_PATTERNS
    )


def pdf_text_from_bytes(
    pdf_bytes_string: bytes,
    max_pages: int = 999,
    pages_sep: str = PAGE_SEP,
) -> str:
    """Convert the PDF byte representation to text."""
    try:
        # Pdfplumber returns empty string for UTF-8 encoded strings
        # (without any exception raised), only Latin-1 works
        # On the other hand, FastAPI requires UTF-8 strings in payloads,
        # so we assume UTF-8 string arrives here
        pdf_bytes_string = pdf_bytes_string.decode("UTF-8").encode("Latin1")
    except UnicodeDecodeError:
        # If the above command fails, we will assume the byte string
        # is already Latin1 encoded
        pass

    all_pages_txt = ""
    all_pages_list = []
    with pdfplumber.open(io.BytesIO(pdf_bytes_string)) as pdf:
        for page_idx in range(
            0, min(len(pdf.pages), max_pages)
        ):  # pylint: disable=invalid-name
            all_pages_txt += pdf.pages[page_idx].extract_text() + pages_sep
            all_pages_list.append(pdf.pages[page_idx].extract_text() + pages_sep)
    return all_pages_list


def clean_text(text):

    if isinstance(text, list):  # Check if input is a list
        text = " ".join(text)  # Join list elements into a single string

    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    return text


# PAGE_SEP constant
PAGE_SEP = "\n" + "=" * 31 + " new page " + "=" * 31 + "\n"


# Function to extract individual pages from the combined text, ensuring page separators are removed.
def extract_pages(all_pages_txt: str) -> list:
    """Extract individual pages from the combined text."""
    # Replace page separator with a unique placeholder
    placeholder = "<PAGE_SEPARATOR>"
    clean_text = all_pages_txt.replace(PAGE_SEP.strip(), placeholder)
    pages = clean_text.split(placeholder)
    pages = [page.strip() for page in pages if page.strip()]
    return pages


# Function for semantic search
def semantic_search(query, model, faiss_index, pages, top_k=5):
    """Performs semantic search to find relevant pages for a given query.

    Args:
        query (str): The search query.
        model: The sentence transformer model used to encode queries.
        faiss_index: The Faiss index to search.
        pages: List of extracted pages from the PDF.
        top_k (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        list: A list of indices of the most similar pages in the Faiss index.
    """
    query_embedding = model.encode([query])[0]  # Generate embedding for the query
    D, I = faiss_index.search(
        query_embedding.reshape(1, -1), top_k
    )  # Search Faiss index

    # Filter indices to be within the valid range of 'pages'
    relevant_page_indices = [index for index in I[0] if 0 <= index < len(pages)]

    return relevant_page_indices


# Handle multiple questions and combine relevant pages
def combined_semantic_search(questions, pages, model, faiss_index, top_k=3):
    """Performs semantic search for multiple questions and combines relevant pages.

    Args:
        questions (list): A list of questions to search for.
        pages: List of extracted pages from the PDF.
        model: The sentence transformer model used to encode queries.
        faiss_index: The Faiss index to search.
        top_k (int, optional): The number of top results to return for each question. Defaults to 3.

    Returns:
        dict: A dictionary where keys are question indices and values are lists of relevant page indices.
    """
    combined_results = {}
    for question_index, question in enumerate(questions):
        relevant_page_indices = semantic_search(
            question, model, faiss_index, pages, top_k
        )
        combined_results[question_index] = relevant_page_indices

    return combined_results


def make_request_mistral(prompt):
    api_key = "nKMmuqxD1WeikspamzeaJRmJOgxBsqsC"
    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return chat_response.choices[0].message.content


# Function to extract data from a document
def extract_data_from_document(path, user_questions, model, faiss_index, pages):
    # Extract individual pages from the cleaned text
    # pages = extract_pages(text)

    # Perform semantic search to find relevant pages
    combined_results = combined_semantic_search(
        user_questions, pages, model, faiss_index
    )

    # Combine relevant pages' text
    relevant_pages_text = ""
    for question_index, relevant_page_indices in combined_results.items():
        for index in relevant_page_indices:
            relevant_pages_text += pages[index] + PAGE_SEP

    # Build the prompt
    pretext = """The following is an extract from a document
    ---- document beginning ----
    """
    posttext = """
    ---- document ending ----
    Please answer the following question about that document :
    """
    user_questions_text = "---- question about the document ----\n\n" + "\n\n".join(
        [f"{i+1}. {q}" for i, q in enumerate(user_questions)]
    )
    prompt = pretext + relevant_pages_text + posttext + user_questions_text

    # Send prompt to the AI
    response = make_request_mistral(prompt)
    return response


def build_faiss_index(pages, model):
    """
    Builds a FAISS index from the given pages and a sentence embedding model.

    Args:
        pages (list of str): The text of the pages extracted from the document.
        model: The sentence transformer model used to embed the pages.

    Returns:
        faiss_index: A FAISS index containing the embeddings.
        page_embeddings: A list of page-level embeddings for future reference.
    """
    # Tokenize and embed each page
    tokenized_pages = []
    for page in pages:
        sentences = page.split(".")  # Simple sentence tokenization
        embeddings = model.encode(sentences)
        tokenized_pages.append(embeddings)

    # Generate page-level embeddings (average sentence embeddings)
    page_embeddings = []
    for page_embedding in tokenized_pages:
        page_embedding_avg = np.mean(
            page_embedding, axis=0
        )  # Aggregate sentence embeddings
        page_embeddings.append(page_embedding_avg)

    # Convert to numpy array for FAISS
    page_embeddings = np.array(page_embeddings)

    # Build the FAISS index with page embeddings
    faiss_index = faiss.IndexFlatL2(page_embeddings.shape[1])  # L2 distance index
    faiss_index.add(page_embeddings)  # Add page embeddings to FAISS

    return faiss_index, page_embeddings


def upload_pdf():
    # Step 1: Upload the PDF
    print("Please upload your PDF file")
    uploaded = files.upload()  # Allows you to upload a file via the Colab interface

    # Step 2: Extract the file name
    # Get the first uploaded file name
    pdf_path = next(iter(uploaded.keys()))
    # Open the PDF and extract text
    with open(pdf_path, "rb") as fobj:
        pdf_bytes_utf8 = fobj.read()

    # Extract text from the PDF bytes
    pdf_text = pdf_text_from_bytes(pdf_bytes_utf8)

    # Extract the text and split it into pages
    text = clean_text(pdf_text)  # Function to clean the text
    pages = extract_pages(text)

    # Build FAISS index
    faiss_index, page_embeddings = build_faiss_index(pages, model)
    return pdf_path, pages, faiss_index


def qrag(pdf_path, pages, faiss_index):

    # Gather user questions dynamically
    while True:
        print("Enter a question you have about your document : ")
        user_questions = []
        question = input("Question: ")
        user_questions.append(question)

        if question == "":
            print("No questions provided. Exiting.")
            return None

        response_document = extract_data_from_document(
            pdf_path, user_questions, model, faiss_index, pages
        )
        print("Response :", response_document)

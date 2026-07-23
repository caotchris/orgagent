import functions_framework
from google.cloud import storage
import pypdf
import io
import os
import psycopg2
from pgvector.psycopg2 import register_vector
import vertexai
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_INSTANCE = os.getenv("DB_INSTANCE")
DB_SOCKET_DIR = os.getenv("DB_SOCKET_DIR", "/cloudsql")

vertexai.init(project=PROJECT_ID, location="us-central1")
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")


def get_conn():
    conn = psycopg2.connect(
        host=f"{DB_SOCKET_DIR}/{DB_INSTANCE}",
        dbname="orgagent", user="postgres", password=DB_PASSWORD,
    )
    register_vector(conn)
    return conn


def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


@functions_framework.cloud_event
def procesar_pdf(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]

    if not file_name.lower().endswith(".pdf"):
        print(f"Ignorando archivo no-PDF: {file_name}")
        return

    print(f"Procesando {file_name}...")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    pdf_bytes = blob.download_as_bytes()

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    chunks = chunk_text(full_text)
    print(f"{len(chunks)} fragmentos generados de {file_name}")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM documents WHERE source_file = %s", (file_name,))

    for chunk in chunks:
        result = embedding_model.get_embeddings([chunk])
        vector = result[0].values
        cur.execute(
            "INSERT INTO documents (content, embedding, source_file) VALUES (%s, %s, %s)",
            (chunk, vector, file_name),
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Listo: {file_name} indexado con {len(chunks)} fragmentos.")
import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from langchain_google_vertexai import VertexAIEmbeddings

load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project=PROJECT_ID)

conn = psycopg2.connect(
    host="127.0.0.1", port=5432, dbname="orgagent", user="postgres",
    password=os.getenv("DB_PASSWORD"),
)
conn.autocommit = True
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(768)
);
""")
register_vector(conn)

sample_docs = [
    "Los voluntarios deben confirmar su asistencia a los eventos con al menos 24 horas de anticipacion llamando al coordinador.",
    "El horario de los eventos de fin de semana es de 9am a 1pm, punto de encuentro en la oficina central.",
    "Para darse de baja como voluntario, hay que enviar un correo a coordinacion@orgagent.org con 48 horas de anticipacion.",
]

for text in sample_docs:
    vector = embeddings.embed_query(text)
    cur.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (text, vector))

print(f"{len(sample_docs)} documentos cargados con embeddings.")
cur.close()
conn.close()
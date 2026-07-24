import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
import vertexai
from vertexai.language_models import TextEmbeddingModel
from fastmcp import FastMCP

load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

mcp = FastMCP("orgagent-tools")

vertexai.init(project=PROJECT_ID, location="us-central1")
_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")


def embed_query(text: str):
    result = _embedding_model.get_embeddings([text])
    return result[0].values

DB_SOCKET_DIR = os.getenv("DB_SOCKET_DIR")
DB_INSTANCE = os.getenv("DB_INSTANCE")


def get_conn():
    if DB_SOCKET_DIR:
        conn = psycopg2.connect(
            host=f"{DB_SOCKET_DIR}/{DB_INSTANCE}",
            dbname="orgagent", user="postgres",
            password=os.getenv("DB_PASSWORD"),
        )
    else:
        conn = psycopg2.connect(
            host="127.0.0.1", port=5432, dbname="orgagent", user="postgres",
            password=os.getenv("DB_PASSWORD"),
        )
    register_vector(conn)
    return conn


UMBRAL_DISTANCIA_RAG = 1.0  # calibrado con datos reales: relevantes ~0.79-0.94, irrelevantes >=1.06

@mcp.tool()
def buscar_documentos(pregunta: str) -> str:
    """Busca informacion relevante en los documentos institucionales para responder la pregunta.
    Devuelve 'NO_ENCONTRADO: ...' si no hay informacion suficientemente relevante."""
    query_embedding = embed_query(pregunta)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT content, source_file, embedding <-> %s::vector AS distancia "
        "FROM documents ORDER BY distancia ASC LIMIT 5",
        (query_embedding,),
    )
    filas = cur.fetchall()
    cur.close()
    conn.close()

    if not filas:
        return "NO_ENCONTRADO: no hay documentos cargados en el sistema."

    mejor_distancia = filas[0][2]
    if mejor_distancia > UMBRAL_DISTANCIA_RAG:
        return (
            "NO_ENCONTRADO: no se encontro informacion suficientemente relevante en los "
            f"documentos institucionales para responder esta pregunta (distancia={mejor_distancia:.3f})."
        )

    partes = []
    for content, source_file, distancia in filas:
        if distancia <= UMBRAL_DISTANCIA_RAG:
            fuente = source_file if source_file else "documento institucional (sin nombre registrado)"
            partes.append(f"[Fuente: {fuente} | relevancia={1 - distancia:.2f}]\n{content}")

    return "\n\n---\n\n".join(partes)


@mcp.tool()
def consultar_usuario(email: str) -> str:
    """Consulta los datos de un usuario por su correo electronico."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT full_name, email, phone, role, status FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return f"No se encontro ningun usuario con el correo {email}."
    full_name, email, phone, role, status = row
    return f"Nombre: {full_name}, Email: {email}, Telefono: {phone}, Rol: {role}, Estado: {status}"


@mcp.tool()
def listar_usuarios_inactivos(minutos: int) -> str:
    """Lista los usuarios activos que no han interactuado en los ultimos N minutos."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT full_name, email FROM users
        WHERE status = 'active' AND last_active_at < NOW() - (%s || ' minutes')::interval
        """,
        (minutos,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        return "No hay usuarios inactivos en ese periodo."
    return "\n".join(f"{name} <{email}>" for name, email in rows)


@mcp.tool()
def crear_recordatorio(email: str, mensaje: str) -> str:
    """Crea una tarea de recordatorio para un usuario, identificado por su correo electronico."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return f"No se encontro ningun usuario con el correo {email}."
    user_id = row[0]
    cur.execute(
        "INSERT INTO tasks (user_id, title, description, status) VALUES (%s, %s, %s, 'pending')",
        (user_id, "Recordatorio automatico", mensaje),
    )
    conn.commit()
    cur.close()
    conn.close()
    return f"Recordatorio creado para {email}: {mensaje}"

@mcp.tool()
def desactivar_usuario_inactivo(email: str, motivo: str) -> str:
    """Desactiva (soft delete) a un usuario inactivo, sin borrar sus datos, y deja registro en audit_log."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return f"No se encontro ningun usuario con el correo {email}."
    user_id = row[0]
    cur.execute("UPDATE users SET status = 'inactivo' WHERE id = %s", (user_id,))
    cur.execute(
        "INSERT INTO audit_log (accion, tabla_afectada, registro_id, detalles, ejecutado_por) "
        "VALUES (%s, %s, %s, %s, %s)",
        ("desactivar_usuario", "users", user_id, motivo, "agente_acciones"),
    )
    conn.commit()
    cur.close()
    conn.close()
    return f"Usuario {email} desactivado (soft delete). Motivo: {motivo}"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port, path="/mcp")
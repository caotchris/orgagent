import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    dbname="orgagent",
    user="postgres",
    password=os.getenv("DB_PASSWORD"),
)
conn.autocommit = True
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    role TEXT DEFAULT 'volunteer',
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    last_active_at TIMESTAMP DEFAULT NOW()
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title TEXT NOT NULL,
    description TEXT,
    due_date DATE,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);
""")

cur.execute("""
INSERT INTO users (full_name, email, phone, role)
VALUES ('Usuario de Prueba', 'prueba@ejemplo.com', '5551234', 'volunteer')
ON CONFLICT (email) DO NOTHING;
""")

print("Tablas creadas correctamente.")
cur.close()
conn.close()
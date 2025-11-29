import psycopg2
from psycopg2.extras import RealDictCursor

def get_connection():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="nodo",
        user="nodo",
        password="nodo_password"
    )
    return conn

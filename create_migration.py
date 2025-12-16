# create_migration.py - Run this once to fix the database
from main import engine, Base
from sqlalchemy import text
import os

print("Starting database migration...")

# Ensure data directory exists
os.makedirs("/opt/render/project/data", exist_ok=True)
os.makedirs("data", exist_ok=True)  # Local fallback

with engine.connect() as conn:
    conn.execute(text("PRAGMA foreign_keys=OFF"))
    
    # Add auth0_sub column if missing
    try:
        conn.execute(text("ALTER TABLE users ADD COLUMN auth0_sub TEXT UNIQUE"))
        print("✓ Added column: auth0_sub")
    except:
        print("→ auth0_sub column already exists")
    
    # Add created_at column if missing
    try:
        conn.execute(text("ALTER TABLE users ADD COLUMN created_at DATETIME"))
        print("✓ Added column: created_at")
    except:
        print("→ created_at column already exists")
    
    conn.execute(text("PRAGMA foreign_keys=ON"))
    conn.commit()

# Ensure tables are up to date
Base.metadata.create_all(bind=engine)
print("Database migration completed successfully!")
print("You can now delete users.db and restart the server.")
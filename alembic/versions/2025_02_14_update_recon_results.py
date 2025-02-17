"""Update recon_results table to use scan_metadata

Revision ID: 2025_02_14_update_recon_results
Revises: ebfc4a20deab
Create Date: 2025-02-14 15:57:48.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2025_02_14_update_recon_results'
down_revision = 'ebfc4a20deab'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Drop the new table if it exists to avoid conflicts
    op.execute("DROP TABLE IF EXISTS recon_results_new")

    op.execute("""
        CREATE TABLE recon_results_new (
            id INTEGER PRIMARY KEY,
            tool TEXT,
            domain TEXT,
            results JSON,
            status TEXT,
            error_message TEXT,
            start_time DATETIME,
            end_time DATETIME,
            project_id INTEGER,
            scan_metadata JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            scan_type TEXT
        )
    """)
    
    # Copy existing data
    op.execute("""
        INSERT INTO recon_results_new (id, tool, domain, results, status, error_message, start_time, end_time, project_id)
        SELECT id, tool, domain, results, status, error_message, start_time, end_time, project_id
        FROM recon_results
    """)
    
    # Drop old table
    op.execute("DROP TABLE recon_results")
    
    # Rename new table
    op.execute("ALTER TABLE recon_results_new RENAME TO recon_results")
    
    # Create index for scan_type
    op.create_index(op.f('ix_recon_results_scan_type'), 
                    'recon_results', ['scan_type'], unique=False)

def downgrade() -> None:
    # Create original table
    op.execute("""
        CREATE TABLE recon_results_old (
            id INTEGER PRIMARY KEY,
            target TEXT,
            data TEXT,
            result_type TEXT
        )
    """)
    
    # Copy data back
    op.execute("""
        INSERT INTO recon_results_old (id, target, data, result_type)
        SELECT id, target, data, result_type
        FROM recon_results
    """)
    
    # Drop new table
    op.execute("DROP TABLE recon_results")
    
    # Rename old table
    op.execute("ALTER TABLE recon_results_old RENAME TO recon_results")

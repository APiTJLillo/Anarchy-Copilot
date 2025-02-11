"""recreate recon results table

Revision ID: ebfc4a20deab
Revises: 
Create Date: 2025-02-09 19:49:40.303781

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ebfc4a20deab'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Check if old table exists and get its data
    old_data = []
    try:
        result = conn.execute(sa.text("SELECT * FROM recon_results"))
        old_data = [dict(row) for row in result]
    except:
        pass

    # Drop any existing new table
    conn.execute(sa.text('DROP TABLE IF EXISTS recon_results_new'))

    # Create new table with updated schema
    op.create_table('recon_results_new',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tool', sa.String(), nullable=True),
        sa.Column('domain', sa.String(), nullable=True),
        sa.Column('results', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('start_time', sa.DateTime(), nullable=True),
        sa.Column('end_time', sa.DateTime(), nullable=True),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create new indexes
    op.create_index(op.f('ix_recon_results_new_domain'), 'recon_results_new', ['domain'], unique=False)
    op.create_index(op.f('ix_recon_results_new_tool'), 'recon_results_new', ['tool'], unique=False)
    op.create_index(op.f('ix_recon_results_new_id'), 'recon_results_new', ['id'], unique=False)
    
    # Insert old data if we had any
    if old_data:
        for row in old_data:
            # Convert row data to new schema
            results = row.get('results', '{}')
            if not results or results == '':
                results = '{}'
            
            conn.execute(
                sa.text("""
                    INSERT INTO recon_results_new 
                    (id, tool, domain, results, project_id, start_time)
                    VALUES (:id, :tool, :domain, :results, :project_id, :timestamp)
                """),
                {
                    'id': row.get('id'),
                    'tool': row.get('tool'),
                    'domain': row.get('domain'),
                    'results': results,
                    'project_id': row.get('project_id'),
                    'timestamp': row.get('timestamp')
                }
            )

    # Drop old table if it exists
    conn.execute(sa.text('DROP TABLE IF EXISTS recon_results'))
    
    # Rename new table
    op.rename_table('recon_results_new', 'recon_results')

    # Drop the temporary indexes
    conn = op.get_bind()
    conn.execute(sa.text('DROP INDEX IF EXISTS ix_recon_results_new_domain'))
    conn.execute(sa.text('DROP INDEX IF EXISTS ix_recon_results_new_tool'))
    conn.execute(sa.text('DROP INDEX IF EXISTS ix_recon_results_new_id'))
    
    # Create final indexes
    op.create_index(op.f('ix_recon_results_domain'), 'recon_results', ['domain'], unique=False)
    op.create_index(op.f('ix_recon_results_tool'), 'recon_results', ['tool'], unique=False)
    op.create_index(op.f('ix_recon_results_id'), 'recon_results', ['id'], unique=False)


def downgrade() -> None:
    # Create old schema table
    op.create_table('recon_results_old',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tool', sa.String(), nullable=True),
        sa.Column('domain', sa.String(), nullable=True),
        sa.Column('results', sa.String(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Copy data back
    conn = op.get_bind()
    conn.execute(sa.text("""
        INSERT INTO recon_results_old (id, tool, domain, results, project_id, timestamp)
        SELECT id, tool, domain, 
               CASE 
                   WHEN results IS NULL THEN ''
                   ELSE json_extract(results, '$') 
               END,
               project_id,
               start_time
        FROM recon_results;
    """))
    
    # Drop new table
    op.drop_table('recon_results')
    
    # Rename old table back
    op.rename_table('recon_results_old', 'recon_results')

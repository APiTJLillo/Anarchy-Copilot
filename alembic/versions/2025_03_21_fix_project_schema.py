"""Fix project schema

Revision ID: 2025_03_21_fix_project_schema
Revises: 2025_03_21_fix_schema
Create Date: 2025-03-21 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '2025_03_21_fix_project_schema'
down_revision = '2025_03_21_fix_schema'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Add is_archived column with default value False
    op.add_column('projects', sa.Column('is_archived', sa.Boolean(), server_default='0', nullable=False))

def downgrade() -> None:
    # Drop is_archived column
    op.drop_column('projects', 'is_archived') 
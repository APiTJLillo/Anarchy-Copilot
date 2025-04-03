"""
Database schema updates for the advanced filtering system.

This module provides the database schema updates needed for the filter system,
including tables for filter rules and settings.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic
revision = 'filter_system_001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create filter_rules table
    op.create_table(
        'filter_rules',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('conditions', sa.Text, nullable=False),  # JSON string
        sa.Column('enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('priority', sa.Integer, nullable=False, default=0),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
        sa.Column('tags', sa.Text, nullable=False),  # JSON string
    )
    
    # Create filter_settings table
    op.create_table(
        'filter_settings',
        sa.Column('key', sa.String(255), primary_key=True),
        sa.Column('value', sa.Text, nullable=True),
    )
    
    # Insert default settings
    op.bulk_insert(
        sa.table(
            'filter_settings',
            sa.column('key', sa.String(255)),
            sa.column('value', sa.Text),
        ),
        [
            {'key': 'mode', 'value': 'ACTIVE'},
        ]
    )

def downgrade():
    # Drop tables
    op.drop_table('filter_settings')
    op.drop_table('filter_rules')

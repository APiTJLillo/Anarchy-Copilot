"""Fix schema issues

Revision ID: 2025_03_21_fix_schema
Revises: 2025_02_17_init
Create Date: 2025-03-21 01:03:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '2025_03_21_fix_schema'
down_revision: str = '2025_02_17_init'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Fix users table
    with op.batch_alter_table('users') as batch_op:
        batch_op.alter_column('hashed_password', new_column_name='password_hash')
        batch_op.add_column(sa.Column('updated_at', sa.DateTime(), nullable=True))

    # Fix projects table
    with op.batch_alter_table('projects') as batch_op:
        batch_op.add_column(sa.Column('settings', sa.JSON(), nullable=True))
        batch_op.drop_column('is_archived')

def downgrade() -> None:
    # Revert projects table changes
    with op.batch_alter_table('projects') as batch_op:
        batch_op.drop_column('settings')
        batch_op.add_column(sa.Column('is_archived', sa.Boolean(), default=False))

    # Revert users table changes
    with op.batch_alter_table('users') as batch_op:
        batch_op.alter_column('password_hash', new_column_name='hashed_password')
        batch_op.drop_column('updated_at') 
"""Initial database setup

Revision ID: 2025_02_17_init
Revises: 
Create Date: 2025-02-17 19:40:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy.sql import func

# revision identifiers, used by Alembic.
revision: str = '2025_02_17_init'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create users table first
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(), nullable=True),
        sa.Column('email', sa.String(), nullable=True),
        sa.Column('hashed_password', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    # Create projects table
    op.create_table(
        'projects',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('scope', sa.JSON(), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('is_archived', sa.Boolean(), default=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_projects_name', 'projects', ['name'], unique=False)

    # Create project_collaborators table
    op.create_table(
        'project_collaborators',
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('project_id', 'user_id')
    )

    # Create recon_modules table
    op.create_table(
        'recon_modules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('is_enabled', sa.Boolean(), default=True),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_run', sa.DateTime(), nullable=True),
        sa.Column('run_frequency', sa.String(), nullable=True),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_recon_modules_name', 'recon_modules', ['name'], unique=False)

    # Create recon_targets table
    op.create_table(
        'recon_targets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('domain', sa.String(), nullable=True),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_scanned', sa.DateTime(), nullable=True),
        sa.Column('scan_frequency', sa.String(), nullable=True),
        sa.Column('target_metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_recon_targets_domain', 'recon_targets', ['domain'], unique=False)

    # Create recon_results table
    op.create_table(
        'recon_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tool', sa.String(), nullable=True),
        sa.Column('target_id', sa.Integer(), nullable=True),
        sa.Column('results', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('start_time', sa.DateTime(), nullable=True),
        sa.Column('end_time', sa.DateTime(), nullable=True),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.Column('scan_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('scan_type', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_recon_results_scan_type', 'recon_results', ['scan_type'], unique=False)
    op.create_index('ix_recon_results_tool', 'recon_results', ['tool'], unique=False)

    # Use batch mode to add foreign key constraint
    with op.batch_alter_table('recon_results', schema=None) as batch_op:
        batch_op.create_foreign_key('fk_recon_results_target_id', 'recon_targets', ['target_id'], ['id'])
    
    # Create vulnerabilities table (missing table - adding it before it's referenced)
    op.create_table(
        'vulnerabilities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('severity', sa.Enum('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO', name='severitylevel'), nullable=True),
        sa.Column('status', sa.Enum('OPEN', 'IN_PROGRESS', 'VERIFIED', 'FIXED', 'WONT_FIX', 'FALSE_POSITIVE', name='vulnerabilitystatus'), 
                  nullable=True, server_default='OPEN'),
        sa.Column('cvss_score', sa.Float(), nullable=True),
        sa.Column('proof_of_concept', sa.Text(), nullable=True),
        sa.Column('steps_to_reproduce', sa.Text(), nullable=True),
        sa.Column('technical_details', sa.Text(), nullable=True),
        sa.Column('recommendation', sa.Text(), nullable=True),
        sa.Column('discovered_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.Column('assigned_user_id', sa.Integer(), nullable=True),
        sa.Column('recon_result_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['assigned_user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.ForeignKeyConstraint(['recon_result_id'], ['recon_results.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_vulnerabilities_title', 'vulnerabilities', ['title'], unique=False)
    
    # Create vulnerability_scans table
    op.create_table(
        'vulnerability_scans',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('scan_type', sa.String(), nullable=True),
        sa.Column('target', sa.String(), nullable=True),
        sa.Column('scanner', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('is_full_scan', sa.Boolean(), default=True),
        sa.Column('start_time', sa.DateTime(), default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('end_time', sa.DateTime(), nullable=True),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_vulnerability_scans_name', 'vulnerability_scans', ['name'], unique=False)

    # Create vulnerability_results table
    op.create_table(
        'vulnerability_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('severity', sa.Enum('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO', name='severitylevel')),
        sa.Column('location', sa.String(), nullable=True),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.Column('confidence', sa.String(), nullable=True),
        sa.Column('cwe_id', sa.String(), nullable=True),
        sa.Column('detected_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('scan_data', sa.JSON(), nullable=True),
        sa.Column('false_positive', sa.Boolean(), default=False),
        sa.Column('ignored', sa.Boolean(), default=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('scan_id', sa.Integer(), nullable=True),
        sa.Column('vulnerability_id', sa.Integer(), nullable=True),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['scan_id'], ['vulnerability_scans.id'], ),
        sa.ForeignKeyConstraint(['vulnerability_id'], ['vulnerabilities.id'], ),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_vulnerability_results_title', 'vulnerability_results', ['title'], unique=False)

    # Create proxy_sessions table
    op.create_table(
        'proxy_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=False),
        sa.Column('settings', sqlite.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_proxy_sessions_id', 'proxy_sessions', ['id'], unique=False)
    op.create_index('ix_proxy_sessions_name', 'proxy_sessions', ['name'], unique=False)

    # Create interception_rules table
    op.create_table(
        'interception_rules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('enabled', sa.Boolean(), default=True),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('conditions', sqlite.JSON(), nullable=False),  # List of {field, operator, value, use_regex}
        sa.Column('action', sa.String(), nullable=False),  # FORWARD, BLOCK, MODIFY
        sa.Column('modification', sqlite.JSON(), nullable=True),  # Headers/body modifications
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['session_id'], ['proxy_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_interception_rules_session_id', 'interception_rules', ['session_id'], unique=False)

    # Create proxy_history table
    op.create_table(
        'proxy_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('method', sa.String(), nullable=False),
        sa.Column('url', sa.String(), nullable=False),
        sa.Column('host', sa.String(), nullable=True),
        sa.Column('path', sa.String(), nullable=True),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('tls_version', sa.String(), nullable=True),
        sa.Column('cipher_suite', sa.String(), nullable=True),
        sa.Column('certificate_info', sqlite.JSON(), nullable=True),
        
        # Request data
        sa.Column('request_headers', sqlite.JSON(), nullable=True),
        sa.Column('request_body', sa.Text(), nullable=True),  # Raw request body (base64 if binary)
        sa.Column('decrypted_request', sa.Text(), nullable=True),  # Decrypted request body
        
        # Response data
        sa.Column('response_headers', sqlite.JSON(), nullable=True),
        sa.Column('response_body', sa.Text(), nullable=True),  # Raw response body (base64 if binary)
        sa.Column('decrypted_response', sa.Text(), nullable=True),  # Decrypted response body
        
        # Metadata
        sa.Column('tags', sqlite.JSON(), nullable=True),  # List of tags (e.g., ["raw", "encrypted", "request", "response"])
        sa.Column('is_intercepted', sa.Boolean(), server_default='1', nullable=False),
        sa.Column('is_encrypted', sa.Boolean(), server_default='0', nullable=False),
        sa.Column('duration', sa.Float(), nullable=True),  # Request duration in seconds
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['proxy_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_proxy_history_session_id', 'proxy_history', ['session_id'], unique=False)
    op.create_index('ix_proxy_history_timestamp', 'proxy_history', ['timestamp'], unique=False)

    # Create tunnel_metrics table
    op.create_table(
        'tunnel_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('connection_id', sa.String(), nullable=True),  # UUID of the connection
        sa.Column('direction', sa.String(), nullable=True),  # "request" or "response"
        sa.Column('bytes_transferred', sa.Integer(), nullable=True),  # Number of bytes in this chunk
        sa.Column('cumulative_bytes', sa.Integer(), nullable=True),  # Running total for this connection direction
        sa.Column('host', sa.String(), nullable=True),  # Target hostname
        sa.Column('port', sa.Integer(), nullable=True),  # Target port
        sa.Column('chunk_number', sa.Integer(), nullable=True),  # Sequential chunk number
        sa.Column('session_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['proxy_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_tunnel_metrics_connection_id', 'tunnel_metrics', ['connection_id'], unique=False)

    # Create proxy_analysis_results table
    op.create_table(
        'proxy_analysis_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('analysis_type', sa.String(), nullable=True),
        sa.Column('findings', sqlite.JSON(), nullable=True),
        sa.Column('severity', sa.String(), nullable=True),
        sa.Column('analysis_metadata', sqlite.JSON(), nullable=True),
        sa.Column('session_id', sa.Integer(), nullable=True),
        sa.Column('history_entry_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['history_entry_id'], ['proxy_history.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['session_id'], ['proxy_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_proxy_analysis_results_analysis_type', 'proxy_analysis_results', ['analysis_type'], unique=False)
    op.create_index('ix_proxy_analysis_results_id', 'proxy_analysis_results', ['id'], unique=False)

def downgrade() -> None:
    op.drop_table('proxy_analysis_results')
    op.drop_table('tunnel_metrics')
    op.drop_table('proxy_history')
    op.drop_table('interception_rules')
    op.drop_table('proxy_sessions')
    op.drop_table('vulnerability_results')
    op.drop_table('vulnerability_scans')
    op.drop_table('vulnerabilities')  # Added this line to drop vulnerabilities table
    op.drop_table('recon_results')
    op.drop_table('recon_targets')
    op.drop_table('recon_modules')
    op.drop_table('project_collaborators')
    op.drop_table('projects')
    op.drop_table('users')

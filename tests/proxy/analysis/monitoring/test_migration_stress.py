"""Stress tests for state migration system."""

import pytest
import multiprocessing as mp
import threading
import random
import sqlite3
import time
import tempfile
from pathlib import Path
import json
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
import queue
import signal

from proxy.analysis.monitoring.state_migration import (
    StateMigrator,
    MigrationInfo,
    migrate_state_db
)

logger = logging.getLogger(__name__)

class StressTester:
    """Run stress tests on migration system."""
    
    def __init__(self, db_path: Path, num_processes: int = 4, num_threads: int = 4):
        self.db_path = db_path
        self.num_processes = num_processes
        self.num_threads = num_threads
        self.stop_event = mp.Event()
        self.error_queue = mp.Queue()
    
    def concurrent_migrations(self, duration: int = 30):
        """Run concurrent migrations."""
        processes = []
        
        for _ in range(self.num_processes):
            p = mp.Process(
                target=self._run_migration_worker,
                args=(duration,)
            )
            p.start()
            processes.append(p)
        
        # Wait for duration
        time.sleep(duration)
        self.stop_event.set()
        
        # Wait for processes
        for p in processes:
            p.join()
        
        # Check for errors
        errors = []
        while not self.error_queue.empty():
            errors.append(self.error_queue.get())
        
        return errors
    
    def _run_migration_worker(self, duration: int):
        """Run migration operations in a worker process."""
        threads = []
        local_errors = queue.Queue()
        
        for _ in range(self.num_threads):
            t = threading.Thread(
                target=self._run_migration_thread,
                args=(local_errors,)
            )
            t.start()
            threads.append(t)
        
        # Wait for duration or stop event
        start_time = time.time()
        while time.time() - start_time < duration and not self.stop_event.is_set():
            time.sleep(0.1)
        
        # Signal threads to stop and wait
        self.stop_event.set()
        for t in threads:
            t.join()
        
        # Transfer errors to main process
        while not local_errors.empty():
            self.error_queue.put(local_errors.get())
    
    def _run_migration_thread(self, error_queue: queue.Queue):
        """Run migration operations in a thread."""
        migrator = StateMigrator(self.db_path)
        
        while not self.stop_event.is_set():
            try:
                # Random migration operation
                operation = random.choice([
                    self._do_migration,
                    self._check_integrity,
                    self._backup_restore,
                    self._modify_data
                ])
                
                operation(migrator)
                
            except Exception as e:
                error_queue.put(f"Migration error: {str(e)}")
                logger.error(f"Migration error: {e}", exc_info=True)
    
    def _do_migration(self, migrator: StateMigrator):
        """Perform migration operation."""
        applied = migrator.migrate()
        if applied and not self.stop_event.is_set():
            success, issues = migrator.verify_integrity()
            if not success:
                raise ValueError(f"Migration integrity check failed: {issues}")
    
    def _check_integrity(self, migrator: StateMigrator):
        """Check database integrity."""
        success, issues = migrator.verify_integrity()
        if not success:
            raise ValueError(f"Integrity check failed: {issues}")
    
    def _backup_restore(self, migrator: StateMigrator):
        """Perform backup and restore."""
        backup_path = migrator.backup_before_migration()
        if backup_path and not self.stop_event.is_set():
            restored = migrator.restore_from_backup(backup_path)
            if not restored:
                raise ValueError("Restore failed")
    
    def _modify_data(self, migrator: StateMigrator):
        """Modify database data."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dashboard_state (
                    user_id, state_type, state_data, version
                ) VALUES (?, ?, ?, ?)
            """, (
                f"user_{random.randint(1, 1000)}",
                "test",
                json.dumps({"test": "data"}),
                "1.0.0"
            ))

@pytest.mark.stress
class TestMigrationStress:
    """Stress tests for migration system."""
    
    @pytest.fixture
    def stress_tester(self) -> StressTester:
        """Create stress tester with temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
            
            # Initialize database
            migrator = StateMigrator(db_path)
            migrator.migrate()
            
            yield StressTester(db_path)
    
    def test_concurrent_migrations(self, stress_tester):
        """Test concurrent migrations under load."""
        errors = stress_tester.concurrent_migrations(duration=30)
        assert not errors, f"Encountered errors: {errors}"
    
    @pytest.mark.parametrize("load_factor", [1, 2, 4, 8])
    def test_migration_under_load(self, stress_tester, load_factor):
        """Test migrations under different load factors."""
        stress_tester.num_processes *= load_factor
        stress_tester.num_threads *= load_factor
        
        errors = stress_tester.concurrent_migrations(duration=10)
        assert not errors, f"Encountered errors under load {load_factor}: {errors}"
    
    def test_interrupted_migration(self, stress_tester):
        """Test handling of interrupted migrations."""
        def interrupt_handler(signum, frame):
            stress_tester.stop_event.set()
        
        # Set up interrupt handler
        original_handler = signal.signal(signal.SIGALRM, interrupt_handler)
        signal.alarm(5)  # Interrupt after 5 seconds
        
        try:
            errors = stress_tester.concurrent_migrations(duration=10)
            assert not errors, f"Encountered errors during interruption: {errors}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    
    def test_migration_recovery(self, stress_tester):
        """Test migration recovery after failures."""
        # Force some failures
        def failing_operation(migrator):
            if random.random() < 0.2:  # 20% failure rate
                raise Exception("Simulated failure")
            stress_tester._do_migration(migrator)
        
        # Replace normal operation with failing one
        original_op = stress_tester._do_migration
        stress_tester._do_migration = failing_operation
        
        try:
            errors = stress_tester.concurrent_migrations(duration=10)
            assert len(errors) > 0, "No failures occurred"
            
            # Verify recovery
            migrator = StateMigrator(stress_tester.db_path)
            success, issues = migrator.verify_integrity()
            assert success, f"Failed to recover: {issues}"
            
        finally:
            stress_tester._do_migration = original_op
    
    def test_large_scale_migration(self, stress_tester):
        """Test migration with large scale data."""
        # Create large dataset
        with sqlite3.connect(str(stress_tester.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stress_test (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
            """)
            
            # Insert large number of rows
            for i in range(100000):
                cursor.execute(
                    "INSERT INTO stress_test (data) VALUES (?)",
                    (f"data_{i}",)
                )
        
        errors = stress_tester.concurrent_migrations(duration=10)
        assert not errors, f"Encountered errors with large dataset: {errors}"
    
    def test_migration_performance(self, stress_tester):
        """Test migration performance under load."""
        start_time = time.time()
        errors = stress_tester.concurrent_migrations(duration=10)
        duration = time.time() - start_time
        
        assert not errors, f"Encountered errors: {errors}"
        assert duration < 15, f"Migration took too long: {duration}s"
    
    @pytest.mark.timeout(60)
    def test_long_running_migration(self, stress_tester):
        """Test long-running migration stability."""
        errors = stress_tester.concurrent_migrations(duration=45)
        assert not errors, f"Encountered errors in long run: {errors}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""Distributed hyperparameter tuning using Optuna."""

import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.integration.ray import RayPruner
from optuna.storages import RDBStorage
import ray
from ray import tune
import numpy as np
import json
from pathlib import Path
import sqlalchemy
from distributed import Client, LocalCluster
import pandas as pd

from .priority_tuning import PriorityTuner, TuningConfig, TuningResult
from .priority_validation import PriorityValidator

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed tuning."""
    n_workers: int = 4  # Number of worker processes
    storage_url: str = "sqlite:///studies.db"  # Storage for sharing trials
    cluster_mode: str = "local"  # local, ray, or dask
    ray_address: Optional[str] = None  # Ray cluster address
    dask_scheduler: Optional[str] = None  # Dask scheduler address
    sync_interval: int = 60  # Sync interval in seconds
    resource_limit: Dict[str, float] = field(default_factory=lambda: {
        "cpu": 0.5,
        "memory": 2048  # MB
    })

class DistributedTuner:
    """Distributed hyperparameter tuning."""
    
    def __init__(
        self,
        tuner: PriorityTuner,
        config: DistributedConfig = None
    ):
        self.tuner = tuner
        self.config = config or DistributedConfig()
        self.storage: Optional[RDBStorage] = None
        self.client: Optional[Any] = None  # Ray/Dask client
        self.studies: Dict[str, optuna.Study] = {}
        
        # Create shared storage
        self._setup_storage()
    
    def _setup_storage(self):
        """Setup shared storage for trials."""
        try:
            self.storage = RDBStorage(
                self.config.storage_url,
                engine_kwargs={"pool_size": self.config.n_workers}
            )
            logger.info("Initialized shared storage")
            
        except Exception as e:
            logger.error(f"Failed to setup storage: {e}")
            raise
    
    async def _setup_cluster(self):
        """Setup distributed compute cluster."""
        if self.config.cluster_mode == "ray":
            try:
                ray.init(
                    address=self.config.ray_address,
                    ignore_reinit_error=True
                )
                self.client = ray
                logger.info(
                    f"Connected to Ray cluster at {self.config.ray_address}"
                )
                
            except Exception as e:
                logger.error(f"Failed to connect to Ray cluster: {e}")
                raise
            
        elif self.config.cluster_mode == "dask":
            try:
                if self.config.dask_scheduler:
                    self.client = Client(self.config.dask_scheduler)
                else:
                    cluster = LocalCluster(
                        n_workers=self.config.n_workers,
                        threads_per_worker=1,
                        memory_limit=f"{self.config.resource_limit['memory']}MB"
                    )
                    self.client = Client(cluster)
                
                logger.info(
                    f"Connected to Dask cluster with {self.config.n_workers} workers"
                )
                
            except Exception as e:
                logger.error(f"Failed to setup Dask cluster: {e}")
                raise
    
    async def _cleanup_cluster(self):
        """Cleanup cluster resources."""
        if self.config.cluster_mode == "ray":
            ray.shutdown()
        elif self.config.cluster_mode == "dask":
            if self.client:
                await self.client.close()
    
    def _create_study(
        self,
        study_name: str
    ) -> optuna.Study:
        """Create distributed study."""
        sampler = TPESampler(n_startup_trials=10)
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
            interval_steps=1
        )
        
        if self.config.cluster_mode == "ray":
            pruner = RayPruner(pruner)
        
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            load_if_exists=True
        )
        
        self.studies[study_name] = study
        return study
    
    async def _run_trial(
        self,
        study_name: str,
        trial_id: int
    ):
        """Run single optimization trial."""
        study = self.studies[study_name]
        trial = study.get_trial(trial_id)
        
        try:
            score = await self.tuner._objective(trial)
            study.tell(trial, score)
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
    
    async def _run_worker(
        self,
        study_name: str,
        worker_id: int
    ):
        """Run worker process."""
        study = self.studies[study_name]
        
        while True:
            try:
                # Get next trial
                trial = study.ask()
                
                # Run trial
                if self.config.cluster_mode == "ray":
                    ray.remote(self._run_trial).options(
                        num_cpus=self.config.resource_limit["cpu"]
                    ).remote(study_name, trial._trial_id)
                    
                elif self.config.cluster_mode == "dask":
                    await self.client.submit(
                        self._run_trial,
                        study_name,
                        trial._trial_id
                    )
                    
                else:
                    await self._run_trial(study_name, trial._trial_id)
                
                # Check if study is complete
                if study.trials_dataframe().shape[0] >= self.tuner.config.n_trials:
                    break
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
            
            # Sync interval
            await asyncio.sleep(self.config.sync_interval)
    
    async def tune_distributed(
        self,
        study_name: Optional[str] = None
    ) -> TuningResult:
        """Run distributed hyperparameter tuning."""
        study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Setup cluster
            await self._setup_cluster()
            
            # Create study
            study = self._create_study(study_name)
            
            # Start workers
            workers = [
                self._run_worker(study_name, i)
                for i in range(self.config.n_workers)
            ]
            
            # Wait for completion
            await asyncio.gather(*workers)
            
            # Get results
            result = await self._collect_results(study)
            
            # Apply best parameters
            await self.tuner._apply_best_params(result.best_params)
            
            return result
            
        finally:
            await self._cleanup_cluster()
    
    async def _collect_results(
        self,
        study: optuna.Study
    ) -> TuningResult:
        """Collect study results."""
        df = study.trials_dataframe()
        
        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            all_trials=[
                {
                    "number": t._trial_id,
                    "params": t.params,
                    "value": t.value if t.value is not None else float("nan")
                }
                for t in study.trials
            ],
            study_info={
                "n_trials": len(study.trials),
                "duration": (
                    study.trials[-1].datetime_complete -
                    study.trials[0].datetime_start
                ),
                "n_workers": self.config.n_workers,
                "cluster_mode": self.config.cluster_mode,
                "state": study.system_attrs
            }
        )
        
        # Export results
        await self.tuner._export_tuning_report(result)
        
        return result
    
    def get_study_status(
        self,
        study_name: str
    ) -> Dict[str, Any]:
        """Get study status."""
        if study_name not in self.studies:
            return {}
        
        study = self.studies[study_name]
        df = study.trials_dataframe()
        
        return {
            "n_trials": len(study.trials),
            "complete_trials": len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
            "running_trials": len(study.get_trials(states=[optuna.trial.TrialState.RUNNING])),
            "failed_trials": len(study.get_trials(states=[optuna.trial.TrialState.FAIL])),
            "best_score": study.best_value if study.best_trial else None,
            "study_duration": (
                study.trials[-1].datetime_complete - study.trials[0].datetime_start
                if study.trials
                else None
            )
        }

def create_distributed_tuner(
    tuner: PriorityTuner,
    config: Optional[DistributedConfig] = None
) -> DistributedTuner:
    """Create distributed tuner."""
    return DistributedTuner(tuner, config)

if __name__ == "__main__":
    # Example usage
    from .priority_tuning import create_priority_tuner
    from .priority_validation import create_priority_validator
    from .adaptive_priority import create_priority_learner
    from .notification_priority import create_priority_router
    from .notification_throttling import create_throttled_manager
    from .notification_channels import create_notification_manager
    
    async def main():
        # Create notification stack
        manager = create_notification_manager()
        throttler = create_throttled_manager(manager)
        router = create_priority_router(throttler)
        learner = create_priority_learner(router)
        validator = create_priority_validator(learner)
        tuner = create_priority_tuner(validator)
        
        # Create distributed tuner
        dist_tuner = create_distributed_tuner(
            tuner,
            DistributedConfig(
                n_workers=4,
                cluster_mode="dask",
                storage_url="sqlite:///tuning_studies.db"
            )
        )
        
        # Run distributed tuning
        result = await dist_tuner.tune_distributed()
        print("Best Parameters:", json.dumps(result.best_params, indent=2))
        print(f"Best Score: {result.best_score:.4f}")
        
        # Check study status
        status = dist_tuner.get_study_status(result.study_info["study_name"])
        print("Study Status:", json.dumps(status, indent=2))
    
    asyncio.run(main())

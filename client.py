"""HTTP client for DataCleaningEnv."""

import httpx
from typing import Optional
from contextlib import asynccontextmanager

from models import CleaningAction, CleaningObservation, CleaningState, ResetRequest


class DataCleaningEnvClient:
    """
    Async HTTP client for interacting with DataCleaningEnv server.
    
    Usage:
        async with DataCleaningEnvClient("http://localhost:7860") as client:
            obs = await client.reset("easy_nulls")
            obs = await client.step(CleaningAction(operation="fill_missing", column="age", value="30"))
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the environment server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
    
    async def reset(self, task_id: str, difficulty: Optional[str] = None) -> CleaningObservation:
        """
        Reset the environment with a new task.
        
        Args:
            task_id: Task identifier (e.g., 'easy_nulls', 'medium_formats', 'hard_multitable')
            difficulty: Optional difficulty level (inferred from task_id if not provided)
            
        Returns:
            Initial observation
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        request = ResetRequest(task_id=task_id, difficulty=difficulty)
        response = await self.client.post(
            f"{self.base_url}/reset",
            json=request.model_dump()
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "observation" in data:
            data = data["observation"]
        return CleaningObservation(**data)
    
    async def step(self, action: CleaningAction) -> CleaningObservation:
        """
        Take a step in the environment.
        
        Args:
            action: Cleaning action to apply
            
        Returns:
            Observation after applying the action
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        response = await self.client.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()}
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "observation" in data:
            data = data["observation"]
        return CleaningObservation(**data)
    
    async def get_state(self) -> CleaningState:
        """
        Get the current environment state.
        
        Returns:
            Current state
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        response = await self.client.get(f"{self.base_url}/state")
        response.raise_for_status()
        return CleaningState(**response.json())
    
    async def list_tasks(self) -> dict:
        """
        List available tasks.
        
        Returns:
            Dictionary of available tasks with metadata
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        response = await self.client.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> dict:
        """
        Check if the server is healthy.
        
        Returns:
            Health status dictionary
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        response = await self.client.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()

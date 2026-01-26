"""
Utility module for parallel OpenAI API calls.

Provides functions for making concurrent API requests with rate limiting
and error handling.
"""

import asyncio
from typing import List, Callable, Any, Optional, Dict, Tuple
from openai import AsyncOpenAI
from tqdm import tqdm


def run_async(coro):
    """
    Run an async coroutine, handling both cases:
    - When no event loop is running: use asyncio.run()
    - When an event loop is already running (e.g., Jupyter): use nest_asyncio or create_task
    
    Args:
        coro: Coroutine to run
    
    Returns:
        Result of the coroutine
    """
    try:
        # Try to get the running event loop
        loop = asyncio.get_running_loop()
        # If we get here, there's a running loop (e.g., in Jupyter)
        # Use nest_asyncio if available, otherwise we need to handle differently
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(coro)
        except ImportError:
            # nest_asyncio not available, create a new event loop in a thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(coro)


async def parallel_api_calls(
    tasks: List[Callable[[], Tuple[List[Dict[str, str]], Dict[str, Any]]]],
    api_key: str,
    model: str,
    max_concurrent: int = 20,
    system_message: Optional[str] = None,
    progress_desc: str = "Processing",
    retry_on_error: bool = True,
    max_retries: int = 3,
    on_complete: Optional[Callable[[int, Any], None]] = None
) -> List[Any]:
    """
    Execute multiple API calls in parallel with rate limiting.
    
    Args:
        tasks: List of callables that return (messages, kwargs) tuples for API calls
        api_key: OpenAI API key
        model: Model to use
        max_concurrent: Maximum number of concurrent requests
        system_message: Optional system message to prepend to all requests
        progress_desc: Description for progress bar
        retry_on_error: Whether to retry failed requests
        max_retries: Maximum number of retries per request
        on_complete: Optional callback function(index, result) called immediately when each result is ready
    
    Returns:
        List of results in the same order as tasks (or Exception objects on error)
    """
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(tasks)
    completed_count = 0
    lock = asyncio.Lock()
    
    async def make_request_with_retry(index: int, task: Callable[[], Tuple[List[Dict[str, str]], Dict[str, Any]]]) -> Tuple[int, Any]:
        """Make API request with retry logic."""
        async with semaphore:
            messages, kwargs = task()
            
            # Add system message if provided
            if system_message:
                messages = [{"role": "system", "content": system_message}] + messages
            
            for attempt in range(max_retries if retry_on_error else 1):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **kwargs
                    )
                    result = response.choices[0].message.content.strip()
                    return (index, result)
                except Exception as e:
                    if attempt < max_retries - 1 and retry_on_error:
                        # Exponential backoff
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        return (index, e)
    
    # Create all tasks
    async_tasks = [
        make_request_with_retry(i, task) for i, task in enumerate(tasks)
    ]
    
    # Execute with progress bar
    pbar = tqdm(total=len(tasks), desc=progress_desc)
    
    async def process_results():
        nonlocal completed_count
        for coro in asyncio.as_completed(async_tasks):
            index, result = await coro
            results[index] = result
            
            # Call callback immediately if provided (before updating progress bar)
            if on_complete:
                try:
                    # Call callback - can be sync or async
                    if asyncio.iscoroutinefunction(on_complete):
                        await on_complete(index, result)
                    else:
                        on_complete(index, result)
                except Exception as e:
                    # Don't let callback errors break the main process
                    print(f"\nWarning: Error in on_complete callback for index {index}: {e}")
            
            async with lock:
                completed_count += 1
                pbar.update(1)
    
    await process_results()
    pbar.close()
    await client.close()
    
    return results


def create_chat_task(
    user_message: str,
    system_message: Optional[str] = None,
    **kwargs
) -> Callable[[], Tuple[List[Dict[str, str]], Dict[str, Any]]]:
    """
    Create a task function for parallel_api_calls.
    
    Args:
        user_message: User message content
        system_message: Optional system message (will be added by parallel_api_calls if provided)
        **kwargs: Additional API call parameters (temperature, response_format, etc.)
    
    Returns:
        Callable that returns (messages, kwargs) tuple
    """
    def task():
        messages = [{"role": "user", "content": user_message}]
        return messages, kwargs
    
    return task

import time
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from google.api_core import exceptions as google_exceptions
import asyncio

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_timestamps = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        if client_ip in self.request_timestamps:
            self.request_timestamps[client_ip] = [t for t in self.request_timestamps[client_ip] if current_time - t < self.window_seconds]
            if len(self.request_timestamps[client_ip]) >= self.max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        self.request_timestamps.setdefault(client_ip, []).append(current_time)
        return await call_next(request)

class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout: float = 30.0):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")

async def google_api_error_handler(request: Request, exc: google_exceptions.GoogleAPIError):
    if isinstance(exc, google_exceptions.ResourceExhausted):
        return JSONResponse(status_code=429, content={"detail": "Google API rate limit exceeded. Please try again later."})
    elif isinstance(exc, google_exceptions.ServiceUnavailable):
        return JSONResponse(status_code=503, content={"detail": "Google API service is currently unavailable. Please try again later."})
    else:
        return JSONResponse(status_code=500, content={"detail": f"An error occurred while processing your request: {str(exc)}"})

def setup_error_handling(app):
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TimeoutMiddleware)
    app.add_exception_handler(google_exceptions.GoogleAPIError, google_api_error_handler)

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": f"An unexpected error occurred: {str(exc)}"}
        )
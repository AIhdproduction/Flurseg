"""Application factory and local server launcher for review."""

from __future__ import annotations

import webbrowser
from hashlib import sha256
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from flurseg import __version__
from flurseg.review.api import build_router
from flurseg.review.service import ReviewService

STATIC_DIR = Path(__file__).with_name("static")


class CacheControlledStaticFiles(StaticFiles):
    """Serve static UI assets with the configured cache policy."""

    def __init__(self, *args: Any, cache_control: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cache_control = cache_control

    async def get_response(self, path: str, scope: Any):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = self.cache_control
        return response


def _static_fingerprint(static_dir: Path) -> str:
    """Return a content fingerprint for the review UI assets."""
    digest = sha256()
    for asset in sorted(path for path in static_dir.rglob("*") if path.is_file()):
        digest.update(asset.relative_to(static_dir).as_posix().encode())
        digest.update(asset.read_bytes())
    return digest.hexdigest()[:12]


def create_app(paths: Any, cfg: Any, backend_factory: Any | None = None) -> FastAPI:
    """Create the offline review UI and JSON API."""
    static_dir = STATIC_DIR
    asset_version = f"{__version__}-{_static_fingerprint(static_dir)}"
    service = ReviewService(paths, cfg, backend_factory)
    app = FastAPI(title="Flurseg Review", docs_url=None, redoc_url=None)
    app.state.review_service = service
    app.state.asset_version = asset_version

    @app.middleware("http")
    async def prevent_api_caching(request: Any, call_next: Any) -> Any:
        """Prevent browser caches from retaining mutable API responses."""
        response = await call_next(request)
        if request.url.path.startswith("/api/") and "/tiles/" not in request.url.path:
            response.headers["Cache-Control"] = "no-store"
        return response

    app.include_router(build_router(service))

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(
            html.replace("__STATIC_VERSION__", asset_version),
            headers={"Cache-Control": "no-cache"},
        )

    app.mount(
        f"/static/{asset_version}",
        CacheControlledStaticFiles(
            directory=static_dir,
            cache_control="public, max-age=31536000, immutable",
        ),
        name="versioned-static",
    )
    app.mount(
        "/static",
        CacheControlledStaticFiles(directory=static_dir, cache_control="no-cache"),
        name="static",
    )
    return app


def run(paths: Any, cfg: Any, backend_factory: Any | None = None, *, port: int | None = None,
        open_browser: bool | None = None) -> None:
    """Run Uvicorn on the configured loopback interface."""
    import uvicorn

    selected_port = port if port is not None else cfg.pipeline.review.port
    should_open = cfg.pipeline.review.open_browser if open_browser is None else open_browser
    url = f"http://{cfg.pipeline.review.host}:{selected_port}"
    if should_open:
        webbrowser.open(url)
    uvicorn.run(create_app(paths, cfg, backend_factory), host=cfg.pipeline.review.host, port=selected_port)

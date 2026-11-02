"""FastAPI routes for the local review application."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from flurseg.review.service import ReviewError, ReviewService


def _error(error: ReviewError, status: int = 400) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=status)


async def _json_body(request: Request) -> dict[str, Any]:
    """Return an optional JSON object, allowing action endpoints without a body."""
    try:
        body = await request.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def build_router(service: ReviewService) -> APIRouter:
    """Build routes without coupling the service layer to FastAPI."""
    router = APIRouter(prefix="/api")

    @router.get("/config")
    def config() -> dict[str, Any]:
        return service.config()

    @router.get("/queue")
    def queue(source: str | None = None, session: str | None = None, reason: str | None = None,
              status: str | None = None) -> list[dict[str, Any]]:
        return service.queue(source=source, session=session, reason=reason, status=status)

    @router.get("/pool")
    def pool() -> list[dict[str, Any]]:
        return service.pool()

    @router.post("/pool/{image_id}/to-review")
    def to_review(image_id: str) -> Response:
        try:
            return JSONResponse(service.to_review(image_id))
        except ReviewError as error:
            return _error(error)

    @router.get("/stats")
    def stats() -> dict[str, Any]:
        return service.stats()

    @router.get("/images/{image_id}")
    def image_meta(image_id: str) -> Response:
        try:
            return JSONResponse(service.image_meta(image_id))
        except ReviewError as error:
            return _error(error, 404)

    @router.get("/images/{image_id}/tiles/{layer}/{level}/{x}/{y}")
    def tile(image_id: str, layer: str, level: int, x: int, y: int) -> Response:
        try:
            path = service.tile(image_id, layer, level, x, y)
            media = "image/jpeg" if path.suffix.lower() == ".jpg" else "image/png"
            return FileResponse(path, media_type=media)
        except (ReviewError, FileNotFoundError, ValueError) as error:
            return _error(ReviewError(str(error)), 404)

    @router.get("/images/{image_id}/mask")
    def mask(image_id: str) -> Response:
        try:
            data, headers = service.mask_bytes(image_id)
            return Response(data, media_type="application/octet-stream", headers={"Content-Encoding": "gzip", **headers})
        except ReviewError as error:
            return _error(error, 404)

    @router.put("/images/{image_id}/mask")
    async def save_mask(image_id: str, request: Request) -> Response:
        try:
            base_version = request.headers.get("X-Proposal-Version") or request.query_params.get("base_proposal_sha256")
            service.save_draft(image_id, await request.body(), base_proposal_sha256=base_version)
            return Response(status_code=204)
        except ReviewError as error:
            return _error(error)

    @router.delete("/images/{image_id}/draft")
    def delete_draft(image_id: str) -> Response:
        try:
            data, headers = service.discard_draft(image_id)
            return Response(data, media_type="application/octet-stream", headers={"Content-Encoding": "gzip", **headers})
        except ReviewError as error:
            return _error(error, 404)

    @router.get("/images/{image_id}/superpixels")
    def superpixels(image_id: str, level: str = "fine") -> Response:
        try:
            return Response(
                service.superpixels_bytes(image_id, level=level),
                media_type="application/octet-stream",
                headers={"Content-Encoding": "gzip"},
            )
        except (ReviewError, FileNotFoundError) as error:
            status = 400 if "unknown superpixel level" in str(error) else 404
            return _error(ReviewError(str(error)), status)

    @router.get("/images/{image_id}/proposal-classes")
    def proposal_classes(image_id: str) -> Response:
        try:
            return Response(service.proposal_classes_bytes(image_id), media_type="application/octet-stream", headers={"Content-Encoding": "gzip"})
        except ReviewError as error:
            return _error(error, 404)

    @router.post("/images/{image_id}/accept")
    async def accept(image_id: str, request: Request) -> Response:
        try:
            body = await _json_body(request)
            return JSONResponse(service.accept(image_id, body.get("review_seconds", 0)))
        except ReviewError as error:
            return _error(error)

    @router.post("/images/{image_id}/reject")
    async def reject(image_id: str, request: Request) -> Response:
        try:
            body = await _json_body(request)
            return JSONResponse(service.reject(image_id, body.get("review_seconds", 0)))
        except ReviewError as error:
            return _error(error)

    @router.post("/images/{image_id}/skip")
    async def skip(image_id: str, request: Request) -> Response:
        try:
            body = await _json_body(request)
            return JSONResponse(service.skip(image_id, body.get("review_seconds", 0)))
        except ReviewError as error:
            return _error(error)

    @router.post("/images/{image_id}/find-similar")
    async def similar(image_id: str, request: Request) -> Response:
        try:
            body = await _json_body(request)
            data, window = service.find_similar_with_window(
                image_id, int(body.get("class_id", -1)), body.get("boxes", [])
            )
            return Response(data, media_type="application/octet-stream", headers={
                "Content-Encoding": "gzip", "X-Window": ",".join(map(str, window)),
            })
        except ReviewError as error:
            status = 503 if "unavailable" in str(error) else 400
            return _error(error, status)

    return router

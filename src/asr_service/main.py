from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .asr_engine import AsrSessionManager, results_to_text

logger = logging.getLogger(__name__)


@dataclass
class AsrError(Exception):
    status_code: int
    error_code: str
    message: str


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


app = FastAPI(default_response_class=UTF8JSONResponse)
manager = AsrSessionManager(timeout_seconds=30)


@app.exception_handler(AsrError)
def handle_asr_error(request, exc: AsrError):
    return UTF8JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "message": exc.message},
    )


@app.post("/asr/start")
def asr_start():
    try:
        manager.start()
    except RuntimeError:
        raise AsrError(400, "InvalidRequest", "ASR session already active")
    except Exception as e:
        logger.exception("ASR start failed")
        raise AsrError(503, "ServiceUnavailable", f"ASR start failed: {e}")

    return {"success": True}


@app.post("/asr/stop")
def asr_stop():
    try:
        results = manager.stop()
    except RuntimeError:
        raise AsrError(400, "AsrNotActive", "ASR session is not active")
    except TimeoutError:
        raise AsrError(408, "ServiceTimeout", "ASR stop timed out")
    except Exception as e:
        logger.exception("ASR stop failed")
        raise AsrError(503, "ServiceUnavailable", f"ASR stop failed: {e}")

    text = results_to_text(results)
    return {"success": True, "text": text}


@app.get("/asr/status")
def asr_status():
    return {"listening": manager.status()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8014)

import hashlib
import json
import time
from typing import Any, Generator, Mapping, Optional

from dify_plugin import Endpoint
from loguru import logger
from pydantic import BaseModel
from werkzeug import Request, Response


class OpenaiCompatibleFiles(Endpoint):
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        app_id: str = settings.get("app_id", {}).get("app_id", "")
        if not app_id:
            raise ValueError("App ID is required")
        if not isinstance(app_id, str):
            raise ValueError("App ID must be a string")

        try:
            data = r.get_json()
            logger.debug(f"收到请求: {json.dumps(data, indent=2, ensure_ascii=False)}")

        except ValueError as e:
            logger.error(f"请求值错误: {str(e)}")
            return Response(f"Error: {e}", status=400, content_type="text/plain")
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            return Response(f"Error: {e}", status=500, content_type="text/plain")

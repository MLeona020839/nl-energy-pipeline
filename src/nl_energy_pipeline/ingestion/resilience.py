import logging
import re
import time
from collections.abc import Callable

import requests

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"(securityToken=)[^&\s]+")


def redact(text: str) -> str:
    """ENTSO-E puts the API token in the query string, and requests echoes the URL in errors."""
    return _TOKEN.sub(r"\1***", text)


def is_transient(exc: Exception) -> bool:
    """Worth retrying: network trouble or a server-side (5xx) failure. A 4xx is our fault."""
    if isinstance(exc, requests.ConnectionError | requests.Timeout):
        return True
    return (
        isinstance(exc, requests.HTTPError)
        and exc.response is not None
        and exc.response.status_code >= 500
    )


def call_with_retries[T](
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `fn`, retrying transient failures with exponential backoff (5s, 10s, 20s ...).

    Anything non-transient (bad token, bad parameters) is raised immediately. When retries
    are exhausted the error is re-raised with the token scrubbed from its message.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except requests.RequestException as exc:
            if not is_transient(exc) and isinstance(exc, requests.HTTPError):
                raise requests.HTTPError(redact(str(exc)), response=exc.response) from None
            if attempt == attempts:
                raise requests.HTTPError(
                    redact(str(exc)), response=getattr(exc, "response", None)
                ) from None
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(
                "transient failure (%s); retry %d/%d in %.0fs",
                redact(str(exc)),
                attempt,
                attempts - 1,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")

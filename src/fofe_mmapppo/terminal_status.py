"""Terminal helper for a compact live training-status line.

The previous implementation used an ANSI scrolling region to pin a status line
at the top of the terminal.  That is not reliable across WSL/Windows Terminal
combinations and can cause existing terminal text to be overwritten.

This version uses the much more portable pattern used by progress bars:
- the live status stays on the current bottom line;
- before a normal log line is printed, the live line is cleared;
- the log line is printed normally;
- the live status is redrawn underneath it.

Redirected/non-TTY output falls back to plain printing without control codes.
"""
from __future__ import annotations

import atexit
import shutil
import sys


class FixedStatusHeader:
    """Maintain one live status line without modifying the terminal scroll region.

    The historical class name is kept to avoid changing callers.  The status is
    intentionally rendered at the bottom/current line rather than literally
    pinned to row 1; this is far more robust in WSL and Windows Terminal.
    """

    def __init__(self) -> None:
        self.enabled = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.cols = max(40, shutil.get_terminal_size(fallback=(120, 30)).columns)
        self._started = False
        self._closed = False
        self._last_text = ""
        self._visible = False

    def _clip(self, text: str) -> str:
        return text[: max(1, self.cols - 1)]

    def _clear_live_line(self) -> None:
        if self.enabled and self._visible:
            # Return to the beginning of the current line and clear it.
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()
            self._visible = False

    def _draw_live_line(self) -> None:
        if not self.enabled or not self._last_text:
            return
        sys.stdout.write("\r\x1b[2K" + self._clip(self._last_text))
        sys.stdout.flush()
        self._visible = True

    def start(self, text: str = "") -> None:
        if self._started:
            if text:
                self.update(text)
            return
        self._started = True
        if self.enabled:
            atexit.register(self.close)
        if text:
            self.update(text)

    def update(self, text: str) -> None:
        self._last_text = text
        if self.enabled:
            self._draw_live_line()

    def log(self, text: str) -> None:
        if self.enabled:
            self._clear_live_line()
            print(text, flush=True)
            self._draw_live_line()
        else:
            print(text, flush=True)

    def plain_status_if_needed(self) -> None:
        """Expose status in redirected logs where an in-place line is impossible."""
        if not self.enabled and self._last_text:
            print(self._last_text, flush=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.enabled and self._visible:
            # Leave the shell prompt on a fresh line when training exits.
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._visible = False

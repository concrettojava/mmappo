"""Small ANSI terminal helper for a fixed training-status header.

Interactive terminals reserve row 1 for a compact live status line while normal
training records scroll below it.  Redirected/non-TTY output falls back to plain
printing without terminal control sequences.
"""
from __future__ import annotations

import atexit
import shutil
import sys


class FixedStatusHeader:
    """Keep a one-line status header fixed at the top of an ANSI terminal."""

    def __init__(self) -> None:
        self.enabled = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.rows = max(3, shutil.get_terminal_size(fallback=(120, 30)).lines)
        self.cols = max(40, shutil.get_terminal_size(fallback=(120, 30)).columns)
        self._started = False
        self._closed = False
        self._last_text = ""

    def start(self, text: str = "") -> None:
        if self._started:
            if text:
                self.update(text)
            return
        self._started = True
        if self.enabled:
            # Reserve row 1. Rows 2..bottom become the scrolling log region.
            sys.stdout.write(f"\x1b[2;{self.rows}r\x1b[2;1H")
            sys.stdout.flush()
            atexit.register(self.close)
        if text:
            self.update(text)

    def update(self, text: str) -> None:
        self._last_text = text
        if not self.enabled:
            return
        # Keep the header within the visible terminal width.
        clipped = text[: max(1, self.cols - 1)]
        # Save cursor, paint row 1, then return to the scrolling region.
        sys.stdout.write(f"\x1b7\x1b[1;1H\x1b[2K{clipped}\x1b8")
        sys.stdout.flush()

    def log(self, text: str) -> None:
        print(text, flush=True)

    def plain_status_if_needed(self) -> None:
        """Expose status in redirected logs where a fixed ANSI header is impossible."""
        if not self.enabled and self._last_text:
            print(self._last_text, flush=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.enabled and self._started:
            # Restore the full scrolling region before returning control to the shell.
            sys.stdout.write("\x1b7\x1b[r\x1b8")
            sys.stdout.flush()

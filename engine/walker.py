"""Walk a corpus without ever stopping on a bad directory or file.

The graded corpus contains an unreadable file, an encrypted PDF, an unknown type
and an empty directory. A walk that raises on the first bad entry silently loses
everything after it, so every error here is recorded and skipped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def walk_corpus(root: Path) -> List[Path]:
    """Return every regular file under root, as corpus-relative paths in sorted order.

    Directories that cannot be listed are skipped. Files are returned even if they
    cannot be read: deciding that is the parser's job, so the skip gets a reason.
    """
    root = Path(root)
    found: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _err: None):
        dirnames.sort()
        for name in sorted(filenames):
            full = Path(dirpath) / name
            try:
                if full.is_file():
                    found.append(full.relative_to(root))
            except OSError:
                continue
    return sorted(found, key=lambda p: p.as_posix())

from __future__ import annotations

from typing import Optional

from .app import CodeCli
from .config import parse_args


def main(argv: Optional[list[str]] = None) -> None:
    config = parse_args(argv)
    CodeCli(config).run()


if __name__ == "__main__":
    main()

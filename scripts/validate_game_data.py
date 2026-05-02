from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.game_data import load


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate extracted game data.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override data root path.")
    args = parser.parse_args()
    data = load(data_root=args.data_root, strict=True)
    print(f"heroes={len(data.heroes)} monsters={len(data.monsters)} tokens={len(data.tokens)} items={len(data.items)} encounters={len(data.encounters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

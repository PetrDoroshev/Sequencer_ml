from __future__ import annotations

import argparse
import json
from pathlib import Path

from miditok import REMI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Папка с tokenizer.json (save_pretrained / from_pretrained).",
    )
    parser.add_argument(
        "--out_txt",
        type=Path,
        default=None,
        help="Куда сохранить id_to_token.txt (по умолчанию рядом с tokenizer.json).",
    )
    parser.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Куда сохранить id_to_token.json (по умолчанию рядом с tokenizer.json).",
    )
    args = parser.parse_args()

    tok_dir: Path = args.tokenizer_dir
    out_txt = args.out_txt or (tok_dir / "id_to_token.txt")
    out_json = args.out_json or (tok_dir / "id_to_token.json")

    tok = REMI.from_pretrained(tok_dir, local_files_only=True)

    # miditok хранит vocab как dict token->id; для C++ удобнее массив id->token.
    token_to_id: dict[str, int] = tok.vocab
    if len(token_to_id) == 0:
        raise RuntimeError("Tokenizer vocab is empty")

    vocab_size = max(token_to_id.values()) + 1
    id_to_token = [""] * vocab_size
    for token, idx in token_to_id.items():
        if idx < 0:
            raise RuntimeError(f"Negative id for token={token!r}: {idx}")
        if idx >= vocab_size:
            raise RuntimeError(f"Out of range id for token={token!r}: {idx}")
        id_to_token[idx] = token

    missing = [i for i, t in enumerate(id_to_token) if t == ""]
    if missing:
        raise RuntimeError(f"Missing tokens for ids: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    out_txt.write_text("\n".join(id_to_token) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(id_to_token, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(id_to_token)} tokens to:\n- {out_txt}\n- {out_json}")


if __name__ == "__main__":
    main()


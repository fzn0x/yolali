# Contributing to yolali

Thanks for considering a contribution. The bar is simple to state: the
library must stay readable in one sitting, and every numerical component
must be tested against values a human computed by hand.

## Ground rules

- **License hygiene.** yolali is Apache-2.0. Do not paste code from AGPL or
  GPL repositories (Ultralytics, the YOLOv9 reference repo, etc.). Reading
  them for understanding is fine; copying is not. Apache-2.0 sources
  (D-FINE, DEIM, RF-DETR) may be adapted with a docstring attribution.
- **Readable over clever.** Small functions, docstrings that spell out
  tensor shapes (`# (B, A, 4) xyxy in pixels`), no one-liners that need a
  whiteboard.
- **Library, not framework.** No global state, no monkey-patching, no
  `sys.argv` parsing under `src/`. CLI entry points live in `scripts/`.
- **Typed.** `pyright --strict` must pass. A `# type: ignore` needs a
  trailing comment explaining why.
- **Tested.** Each module ships with pytest tests. Losses, assigners, box
  ops, NMS and mAP get hand-computed expected values, not just shape checks.

## Setup

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

## Commits

We use [Conventional Commits](https://www.conventionalcommits.org/) and
prefer small commits that each do one thing:

```
feat(loss): add varifocal classification loss
fix(assign): clamp alignment metric before power
test(ops): hand-computed CIoU cases
docs(design): record DFL decode deviation
```

Scopes mirror the package layout: `ops`, `assign`, `loss`, `nn`, `data`,
`train`, `eval`, `infer`, `export`, `zoo`, `docs`, `ci`.

## Pull requests

1. Open an issue first for anything beyond a bug fix.
2. Keep PRs focused. One phase of the plan is already too big for one PR.
3. CI must be green: lint, type check, unit tests on CPU.
4. If you touch a numerical component, update or add a hand-computed test.
5. If you make an architectural decision that departs from a cited paper,
   record it under "Deviations" in `docs/design.md`.

## Reporting a benchmark

Never report a metric you did not run. `BENCHMARKS.md` entries need the
exact command, the commit hash, the hardware, and the wall-clock time.

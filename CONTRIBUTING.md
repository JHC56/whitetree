# Contributing

Bug reports and pull requests are welcome. For new features, please open an issue first
so we can discuss it.

## Setup

```
git clone https://github.com/whitetree-dev/whitetree.git
cd whitetree
pip install -e .[test]
```

## Before sending a pull request

```
ruff check src tests benchmarks examples
python tests/test_dynamic_kdtree.py
python tests/test_concurrency.py
```

The tests compare every answer against a static `scipy.spatial.cKDTree`. If a change makes
any distance differ, it is a bug.

## Style

PEP 8, 100 columns. Keep comments short.

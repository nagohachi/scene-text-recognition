# scene-text-recognition

## Installation

```sh
uv sync
```

## Usage

### Testing

```sh
# Run all tests
pytest -v

# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only slow tests (e.g., MNIST training)
pytest -m slow -v -s
```

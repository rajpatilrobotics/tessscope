"""Small project smoke entry point."""

from tessscope import __version__


def main() -> None:
    """Print a stable smoke-test message."""
    print(f"TessScope {__version__}: environment ready")


if __name__ == "__main__":
    main()

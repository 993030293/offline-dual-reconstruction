"""Bind only the exact reduction in the preserved complete native RPC worker."""

import integer_sift_worker as native
from fused_integer_reduction import nearest_many


def main():
    original = native.nearest_many
    native.nearest_many = nearest_many
    try:
        native.main()
    finally:
        native.nearest_many = original


if __name__ == "__main__":
    main()

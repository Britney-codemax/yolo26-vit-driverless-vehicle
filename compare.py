"""Summarize saved results only: no torch imports, training, inference or downloads."""
from fusion.records import refresh


def main():
    rows, groups = refresh()
    print(f"Updated docs/EXPERIMENTS.md: {len(rows)} runs, {len(groups)} comparable groups")


if __name__ == "__main__":
    main()

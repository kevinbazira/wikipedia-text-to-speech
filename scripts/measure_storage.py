#!/usr/bin/env python3
"""Measure storage footprint of generated audio files.

Usage::

    python3 scripts/measure_storage.py [--dir audio_output]
"""

import argparse
import os
from collections import defaultdict

KB = 1024
MB = 1024 * KB
GB = 1024 * MB


def fmt_bytes(b: int) -> str:
    if b >= GB:
        return f"{b / GB:.1f} GB"
    if b >= MB:
        return f"{b / MB:.1f} MB"
    return f"{b / KB:.1f} KB"


def measure(root: str) -> None:
    mp3_sizes: list[int] = []
    vtt_sizes: list[int] = []
    per_article: dict[str, dict] = defaultdict(lambda: {"mp3": 0, "vtt": 0, "sections": 0})

    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            size = os.path.getsize(fpath)
            article = os.path.basename(dirpath)

            if fname.endswith(".mp3"):
                mp3_sizes.append(size)
                per_article[article]["mp3"] += size
                per_article[article]["sections"] += 1
            elif fname.endswith(".vtt"):
                vtt_sizes.append(size)
                per_article[article]["vtt"] += size

    total_mp3 = sum(mp3_sizes)
    total_vtt = sum(vtt_sizes)
    article_count = len(per_article)
    section_count = sum(a["sections"] for a in per_article.values())

    print("=" * 60)
    print("PER-FILE STATISTICS")
    print("=" * 60)
    _print_stats("MP3 files", mp3_sizes)
    _print_stats("VTT files", vtt_sizes)

    print()
    print("=" * 60)
    print("PER-ARTICLE STATISTICS")
    print("=" * 60)
    article_totals = [a["mp3"] + a["vtt"] for a in per_article.values()]
    article_mp3 = [a["mp3"] for a in per_article.values()]
    _print_stats("Total per article (MP3 + VTT)", article_totals)
    _print_stats("MP3 only per article", article_mp3)

    print()
    print("=" * 60)
    print("AGGREGATES")
    print("=" * 60)
    print(f"  Articles found:     {article_count}")
    print(f"  Sections found:     {section_count}")
    print(f"  Avg sections/art:   {section_count / article_count:.1f}" if article_count else "  N/A")
    print(f"  Total MP3:          {fmt_bytes(total_mp3)}")
    print(f"  Total VTT:          {fmt_bytes(total_vtt)}")
    print(f"  Total combined:     {fmt_bytes(total_mp3 + total_vtt)}")

    if article_count > 0:
        top = sorted(per_article.items(), key=lambda x: x[1]["mp3"], reverse=True)[:5]
        print()
        print("Top 5 articles by MP3 size:")
        for name, info in top:
            print(f"  {name}: {fmt_bytes(info['mp3'])} ({info['sections']} sections)")

    # ── Projection ──
    if article_count > 0:
        avg_mp3_per_sec = total_mp3 / section_count if section_count else 0
        avg_vtt_per_sec = total_vtt / section_count if section_count else 0
        avg_size_per_article = (total_mp3 + total_vtt) / article_count
        print()
        print("=" * 60)
        print("PROJECTIONS (for planning)")
        print("=" * 60)
        print(f"  Avg MP3 per section:     {fmt_bytes(int(avg_mp3_per_sec))}")
        print(f"  Avg VTT per section:     {fmt_bytes(int(avg_vtt_per_sec))}")
        print(f"  Avg total per article:   {fmt_bytes(int(avg_size_per_article))}")
        print(f"  Est. 7,000 articles:     {fmt_bytes(int(avg_size_per_article * 7000))}")
        print(f"  Est. 10,000 articles:    {fmt_bytes(int(avg_size_per_article * 10000))}")


def _print_stats(label: str, sizes: list[int]) -> None:
    if not sizes:
        print(f"  {label}: (no data)")
        return
    sorted_sizes = sorted(sizes)
    n = len(sorted_sizes)
    print(f"  {label}:")
    print(f"    Count:  {n}")
    print(f"    Min:    {fmt_bytes(sorted_sizes[0])}")
    print(f"    P50:    {fmt_bytes(sorted_sizes[n // 2])}")
    print(f"    P95:    {fmt_bytes(sorted_sizes[int(n * 0.95)])}")
    print(f"    P99:    {fmt_bytes(sorted_sizes[int(n * 0.99)])}")
    print(f"    Max:    {fmt_bytes(sorted_sizes[-1])}")
    print(f"    Avg:    {fmt_bytes(int(sum(sorted_sizes) / n))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure TTS audio storage footprint")
    parser.add_argument("--dir", default="audio_output", help="Audio output directory (default: audio_output)")
    args = parser.parse_args()
    measure(args.dir)

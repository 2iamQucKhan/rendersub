#!/usr/bin/env python3
import argparse
import os
from transcriber import parse_srt_string, segments_to_srt, merge_overlapping_strings, is_similar


def merge_segments(all_segments, gap_tolerance=0.08, strategy='combine'):
    """
    all_segments: list of segments dicts with 'start','end','text'
    strategy: 'combine' (smart merge using merge_overlapping_strings), 'prefer_first', 'prefer_longest'
    gap_tolerance: seconds tolerance to merge adjacent segments
    """
    if not all_segments:
        return []
    # Normalize and sort
    segs = [dict(s) for s in all_segments]
    segs.sort(key=lambda s: (s.get('start', 0.0), -(s.get('end', 0.0) - s.get('start', 0.0))))

    merged = []
    cur = segs[0].copy()
    for nxt in segs[1:]:
        # If next starts before current ends + tolerance -> consider overlap/adjacent
        if nxt['start'] <= cur['end'] + gap_tolerance:
            # overlapping or adjacent
            # choose merge behavior based on similarity
            if is_similar(cur.get('text',''), nxt.get('text','')):
                # very similar -> extend end and prefer longer text
                cur['end'] = max(cur['end'], nxt['end'])
                if len(nxt.get('text','')) > len(cur.get('text','')):
                    cur['text'] = nxt.get('text','')
            else:
                if strategy == 'combine':
                    merged_text = merge_overlapping_strings(cur.get('text',''), nxt.get('text',''))
                    cur['text'] = merged_text
                    cur['end'] = max(cur['end'], nxt['end'])
                elif strategy == 'prefer_first':
                    cur['end'] = max(cur['end'], nxt['end'])
                elif strategy == 'prefer_longest':
                    # pick text with longer length
                    if len(nxt.get('text','')) > len(cur.get('text','')):
                        cur['text'] = nxt.get('text','')
                    cur['end'] = max(cur['end'], nxt['end'])
                else:
                    # default combine
                    cur['text'] = merge_overlapping_strings(cur.get('text',''), nxt.get('text',''))
                    cur['end'] = max(cur['end'], nxt['end'])
        else:
            # Not overlapping -> push current and move on
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)

    # Post-process: remove empty texts and trim tiny segments
    cleaned = []
    for s in merged:
        txt = (s.get('text') or '').strip()
        dur = s.get('end', 0.0) - s.get('start', 0.0)
        if not txt:
            continue
        if dur <= 0.05:
            # skip almost empty duration
            continue
        cleaned.append({'start': s['start'], 'end': s['end'], 'text': txt})
    return cleaned


def merge_files(input_files, out_path, strategy='combine', gap_tolerance=0.08):
    all_segments = []
    for f in input_files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                text = fh.read()
        except Exception:
            # try fallback encodings
            with open(f, 'r', encoding='latin-1', errors='ignore') as fh:
                text = fh.read()
        segs = parse_srt_string(text)
        all_segments.extend(segs)

    merged = merge_segments(all_segments, gap_tolerance=gap_tolerance, strategy=strategy)
    # sort and renumber
    merged.sort(key=lambda s: s['start'])
    srt_text = segments_to_srt(merged)
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write(srt_text)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Gộp nhiều file SRT thành 1 SRT thông minh")
    p.add_argument('inputs', nargs='+', help='Các file SRT cần gộp')
    p.add_argument('-o', '--output', help='File SRT đầu ra', required=True)
    p.add_argument('--strategy', choices=['combine','prefer_first','prefer_longest'], default='combine')
    p.add_argument('--gap', type=float, default=0.08, help='Ngưỡng ghép kề nhau (giây)')
    args = p.parse_args()

    out = merge_files(args.inputs, args.output, strategy=args.strategy, gap_tolerance=args.gap)
    print(f"Đã gộp xong: {out}")

if __name__ == '__main__':
    main()

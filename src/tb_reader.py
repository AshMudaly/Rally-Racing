"""
tb_reader.py — minimal, dependency-free reader for TensorBoard event files.

Stable-Baselines3 writes scalar metrics (rollout/ep_rew_mean, train/loss, …) as
TensorBoard ``*.tfevents.*`` files in each run's log directory. The control
GUI plots those scalars live without requiring the (heavy) tensorboard or
tensorflow packages: this module parses the event-file format directly.

Event-file format (TFRecord framing):
    each record =  uint64 length            (little-endian)
                   uint32 masked-crc32 of length
                   <length> bytes of payload (a serialized Event protobuf)
                   uint32 masked-crc32 of payload

We only need scalar summaries, so rather than depend on a protobuf library we
parse the handful of protobuf wire-format fields that carry
``Event.step``, ``Event.wall_time`` and ``Event.summary.value[].{tag,simple_value}``.
This is intentionally small and tolerant: anything it can't parse is skipped.

Public API:
    read_scalars(path) -> dict[str, list[(step, wall_time, value)]]
    latest_event_file(run_dir) -> str | None
"""

import os
import struct


# ── protobuf wire-format helpers ─────────────────────────────────────────────
def _read_varint(buf, i):
    """Decode a base-128 varint at offset i; return (value, new_offset)."""
    result = 0
    shift = 0
    while True:
        if i >= len(buf):
            raise IndexError("varint truncated")
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, i


def _skip_field(buf, i, wire_type):
    """Advance past a field whose value we don't care about."""
    if wire_type == 0:          # varint
        _, i = _read_varint(buf, i)
    elif wire_type == 1:        # 64-bit
        i += 8
    elif wire_type == 2:        # length-delimited
        ln, i = _read_varint(buf, i)
        i += ln
    elif wire_type == 5:        # 32-bit
        i += 4
    else:
        raise ValueError(f"unknown wire type {wire_type}")
    return i


def _parse_tensor_proto(buf):
    r"""Extract a single float from a scalar ``TensorProto``.

    Modern TensorBoard/torch writers store scalars as a ``tensor`` field rather
    than the legacy ``simple_value``. For a scalar the float lives either in the
    packed ``tensor_content`` (field 4, raw little-endian float32 bytes) or in
    the repeated ``float_val`` (field 5). Returns the float, or ``None`` if no
    scalar value is found.
    """
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wt = key >> 3, key & 0x07
        if field == 4 and wt == 2:          # tensor_content (bytes)
            ln, i = _read_varint(buf, i)
            content = buf[i:i + ln]
            i += ln
            if len(content) >= 4:
                return struct.unpack("<f", content[:4])[0]
        elif field == 5 and wt == 5:        # float_val (repeated float32)
            val = struct.unpack("<f", buf[i:i + 4])[0]
            i += 4
            return val
        elif field == 6 and wt == 5:        # double_val (repeated, some writers)
            val = struct.unpack("<f", buf[i:i + 4])[0]
            i += 4
            return val
        else:
            i = _skip_field(buf, i, wt)
    return None


def _parse_summary_value(buf):
    """Parse one summary.Value submessage -> (tag, value) or None.

    Handles both the legacy ``simple_value`` (field 3) and the modern
    ``tensor`` (field 8) scalar encodings, so logs from any SB3 / TensorBoard
    version are read correctly.
    """
    i = 0
    tag = None
    value = None
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wt = key >> 3, key & 0x07
        if field == 1 and wt == 2:          # tag (string)
            ln, i = _read_varint(buf, i)
            tag = buf[i:i + ln].decode("utf-8", "replace")
            i += ln
        elif field == 2 and wt == 2:        # node_name (skip)
            ln, i = _read_varint(buf, i)
            i += ln
        elif field == 3 and wt == 5:        # simple_value (float32, legacy)
            value = struct.unpack("<f", buf[i:i + 4])[0]
            i += 4
        elif field == 8 and wt == 2:        # tensor (TensorProto, modern)
            ln, i = _read_varint(buf, i)
            tv = _parse_tensor_proto(buf[i:i + ln])
            if tv is not None:
                value = tv
            i += ln
        else:
            i = _skip_field(buf, i, wt)
    if tag is not None and value is not None:
        return tag, value
    return None


def _parse_summary(buf):
    """Parse a Summary message -> list of (tag, value)."""
    out = []
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wt = key >> 3, key & 0x07
        if field == 1 and wt == 2:          # repeated Value value = 1
            ln, i = _read_varint(buf, i)
            parsed = _parse_summary_value(buf[i:i + ln])
            if parsed:
                out.append(parsed)
            i += ln
        else:
            i = _skip_field(buf, i, wt)
    return out


def _parse_event(buf):
    """Parse an Event message -> (step, wall_time, [(tag, value), …])."""
    i = 0
    step = 0
    wall_time = 0.0
    scalars = []
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wt = key >> 3, key & 0x07
        if field == 1 and wt == 1:          # wall_time (double)
            wall_time = struct.unpack("<d", buf[i:i + 8])[0]
            i += 8
        elif field == 2 and wt == 0:        # step (int64)
            step, i = _read_varint(buf, i)
        elif field == 5 and wt == 2:        # summary (Summary message)
            ln, i = _read_varint(buf, i)
            scalars = _parse_summary(buf[i:i + ln])
            i += ln
        else:
            i = _skip_field(buf, i, wt)
    return step, wall_time, scalars


# ── TFRecord framing ─────────────────────────────────────────────────────────
def _iter_tfrecords(data):
    """Yield payload bytes for each TFRecord in the file content."""
    i = 0
    n = len(data)
    while i + 12 <= n:
        length = struct.unpack("<Q", data[i:i + 8])[0]
        i += 12  # 8-byte length + 4-byte length-CRC (CRC not verified)
        if i + length + 4 > n:
            break  # truncated final record (e.g. still being written)
        payload = data[i:i + length]
        i += length + 4  # payload + 4-byte payload-CRC
        yield payload


# ── public API ───────────────────────────────────────────────────────────────
def read_scalars(path):
    r"""Parse one event file into ``{tag: [(step, wall_time, value), ...]}``.

    Iterates the file's TFRecords and decodes each ``Event`` protobuf, keeping
    only scalar summaries. The result maps each metric tag (e.g.
    ``rollout/ep_rew_mean``) to its time series of triples
    :math:`(s_i, t_i, x_i)` — training step, wall-clock time, and value — in
    file order. Malformed or partially-written records are skipped, so the
    function is safe to call on a file SB3 is still appending to.
    """
    series = {}
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return series
    for payload in _iter_tfrecords(data):
        try:
            step, wt, scalars = _parse_event(payload)
        except (IndexError, ValueError, struct.error):
            continue
        for tag, value in scalars:
            series.setdefault(tag, []).append((step, wt, value))
    return series


def latest_event_file(run_dir):
    r"""Return the most-recently-modified tfevents file at or below ``run_dir``.

    Searches recursively, because Stable-Baselines3 / PPO writes its events into
    an auto-created numbered subdirectory (e.g. ``<run>/PPO_1/events.out...``)
    rather than directly in the run directory. Walking the tree means the panel
    finds the logs whether they sit in the run folder itself or one level down.
    """
    if not os.path.isdir(run_dir):
        return None
    candidates = []
    for root, _dirs, files in os.walk(run_dir):
        for name in files:
            if "tfevents" in name:
                full = os.path.join(root, name)
                try:
                    candidates.append((os.path.getmtime(full), full))
                except OSError:
                    pass
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def read_run(run_dir):
    """Convenience: read scalars from the latest event file in a run dir."""
    ev = latest_event_file(run_dir)
    return read_scalars(ev) if ev else {}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        s = read_run(sys.argv[1])
        for tag, pts in sorted(s.items()):
            print(f"{tag:40s} {len(pts):5d} points  last={pts[-1][2]:.3f}")
    else:
        print("usage: python3 tb_reader.py <run_log_dir>")
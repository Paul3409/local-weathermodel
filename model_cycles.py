"""Model initialization cycles shown only after a conservative publication delay.

UTC cycle labels are initialization times, never local wall-clock times.
The four entries are the four most recent *eligible* cycles, not a promise
that every forecast step or parameter is already present on the server.
"""
from datetime import datetime, timedelta, timezone

# Publication buffers are deliberately conservative. A cycle is not ready
# just because its nominal initialization hour has passed.
PUBLICATION_DELAY_HOURS = {
    "ICON-D2": 3,
    "ICON-EU": 4,
    "GFS": 5,
    "GEFS": 6,
    "GEFS Mittel": 6,
    "ECMWF IFS": 7,
    "ECMWF ENS": 8,
    "AIGFS": 6,
    "AIGEFS Mittel": 7,
}

def recent_cycles(model, now=None, count=4):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    cutoff = now.astimezone(timezone.utc) - timedelta(
        hours=PUBLICATION_DELAY_HOURS.get(model, 6)
    )
    cutoff = cutoff.replace(minute=0, second=0, microsecond=0)
    latest = cutoff.replace(hour=(cutoff.hour // 6) * 6)
    return [latest - timedelta(hours=6 * index) for index in range(count)]

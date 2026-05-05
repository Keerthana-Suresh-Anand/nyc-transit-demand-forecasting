"""
Ingests NYC major-venue events into S3 bronze/events/.

Primary:  Ticketmaster Discovery API (TICKETMASTER_API_KEY)
Fallback: NYC Open Data Permitted Event Information (no key required)

Writes: bronze/events/events_{today}.json  — list of {name, date, venue, source}
"""
from datetime import date, timedelta

import requests

from src.utils.config import (
    S3_EVENTS_PREFIX,
    TICKETMASTER_API_KEY,
)
from src.utils.logger import get_logger
from src.utils.s3_helpers import get_s3_client, write_s3_json

logger = get_logger(__name__)

NYC_VENUES = {
    "KovZpZAE6lFX": "Madison Square Garden",
    "KovZpZAJ6l1A": "Barclays Center",
    "KovZpZAJFe7A": "Yankee Stadium",
    "KovZpZAJaFaA": "Citi Field",
    "KovZpZAeJeaA": "MetLife Stadium",
}

TICKETMASTER_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
NYC_OPEN_DATA_URL = (
    "https://data.cityofnewyork.us/resource/tvpp-9vvx.json"
    "?$limit=500&$order=start_date_time ASC"
)


def _fetch_ticketmaster(start_date: date, end_date: date) -> list[dict]:
    if not TICKETMASTER_API_KEY:
        logger.warning("TICKETMASTER_API_KEY not set — skipping Ticketmaster fetch")
        return []

    events: list[dict] = []
    for venue_id, venue_name in NYC_VENUES.items():
        params = {
            "apikey": TICKETMASTER_API_KEY,
            "venueId": venue_id,
            "startDateTime": f"{start_date}T00:00:00Z",
            "endDateTime": f"{end_date}T23:59:59Z",
            "size": 50,
        }
        try:
            resp = requests.get(TICKETMASTER_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("_embedded", {}).get("events", [])
            for item in items:
                event_date = (item.get("dates", {}).get("start", {}).get("localDate") or "")
                events.append({
                    "name": item.get("name", ""),
                    "date": event_date,
                    "venue": venue_name,
                    "source": "ticketmaster",
                })
            logger.info(f"Ticketmaster: {len(items)} events at {venue_name}")
        except Exception as e:
            logger.warning(f"Ticketmaster fetch failed for {venue_name}: {e}")

    return events


def _fetch_nyc_open_data(start_date: date, end_date: date) -> list[dict]:
    """Fallback: NYC Permitted Event Information (public, no key required)."""
    try:
        resp = requests.get(
            NYC_OPEN_DATA_URL
            + f"&$where=start_date_time>='{start_date}T00:00:00' AND start_date_time<='{end_date}T23:59:59'",
            timeout=20,
        )
        resp.raise_for_status()
        items = resp.json()
        events = []
        for item in items:
            event_date = (item.get("start_date_time") or "")[:10]
            events.append({
                "name": item.get("event_name", item.get("event_type", "NYC Event")),
                "date": event_date,
                "venue": item.get("event_location", ""),
                "source": "nyc_open_data",
            })
        logger.info(f"NYC Open Data: {len(events)} permitted events")
        return events
    except Exception as e:
        logger.warning(f"NYC Open Data fetch failed: {e}")
        return []


def run() -> None:
    logger.info("Starting events ingestion")
    today = date.today()
    start = today - timedelta(days=7)
    end = today + timedelta(days=30)

    events = _fetch_ticketmaster(start, end)

    if not events:
        logger.info("Ticketmaster returned no results — trying NYC Open Data fallback")
        events = _fetch_nyc_open_data(start, end)

    if not events:
        logger.info("No events found from any source — skipping S3 write")
        return

    # Deduplicate by (name, date)
    seen: set[tuple] = set()
    unique: list[dict] = []
    for ev in events:
        key = (ev["name"], ev["date"])
        if key not in seen:
            seen.add(key)
            unique.append(ev)

    s3 = get_s3_client()
    s3_key = f"{S3_EVENTS_PREFIX}events_{today}.json"
    write_s3_json(s3, unique, s3_key)
    logger.info(f"Events ingestion complete: {len(unique)} unique events written to {s3_key}")


if __name__ == "__main__":
    run()

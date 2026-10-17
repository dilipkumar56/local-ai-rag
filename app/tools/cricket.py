"""Fetch cricket statistics as structured table data."""

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (local-ai-rag/1.0)"}


def fetch_stat_tables(url: str, timeout: int = 10) -> list[list[list[str]]]:
    """Return HTML tables as rows of cell text and fail clearly on HTTP errors."""
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return [
        [
            [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            for row in table.find_all("tr")
            if row.find_all(["th", "td"])
        ]
        for table in soup.find_all("table")
    ]


def main() -> None:
    url = "https://www.cricbuzz.com/cricket-stats/player/576/rohit-sharma"
    tables = fetch_stat_tables(url)
    print(f"Number of tables: {len(tables)}")
    for index, table in enumerate(tables, start=1):
        print(f"\n========== TABLE {index} ==========")
        for row in table:
            print(row)


if __name__ == "__main__":
    main()

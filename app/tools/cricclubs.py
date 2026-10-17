"""Small HTTP client for CricClubs profile pages."""

import requests

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (local-ai-rag/1.0)"}


def fetch_profile_html(url: str, timeout: int = 15) -> str:
    """Fetch profile HTML and raise a useful error if the site rejects the request."""
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def main() -> None:
    url = "https://cricclubs.com/LansingCricketLeagueLCL/user/GzmneeKMSIAGdfMUCnsNGg?playerName=Salman+Salarzai"
    html = fetch_profile_html(url)
    print(f"Fetched {len(html)} characters.")


if __name__ == "__main__":
    main()

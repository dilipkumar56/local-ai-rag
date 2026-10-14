import requests
from bs4 import BeautifulSoup

url = "https://www.cricbuzz.com/cricket-stats/player/576/rohit-sharma"

response = requests.get(
    url,
    headers={
        "User-Agent": "Mozilla/5.0"
    },
    timeout=10
)

print("Status:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

tables = soup.find_all("table")

print("Number of tables:", len(tables))

for index, table in enumerate(tables):
    print(f"\n========== TABLE {index} ==========")

    rows = table.find_all("tr")

    for row in rows:
        cells = row.find_all(["th", "td"])

        values = [
            cell.get_text(" ", strip=True)
            for cell in cells
        ]

        if values:
            print(values)
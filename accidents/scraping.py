"""HTML parsing for the politiaromana.ro "Info trafic" pages.

Kept separate from the Selenium driver so it can be tested on saved HTML.
"""
from bs4 import BeautifulSoup

BASE_URL = "https://politiaromana.ro/ro/info-trafic?page={page}&statusActiv=0&categorie=2&data="
CSV_HEADER = ["Nr.crt.", "Titlu", "Data", "Ora", "Descriere", "Reluare"]

# Paragraph positions inside div.descDetaliiObiecte
P_DATE, P_TIME, P_DESCRIPTION, P_RESUMPTION = 0, 1, 3, 4


def parse_listing(html):
    """Article URLs on a listing page (empty list = no more pages)."""
    soup = BeautifulSoup(html, "html.parser")
    return [a["href"] for a in soup.select("h3 a") if a.get("href")]


def parse_article(html):
    """Return dict(Titlu, Data, Ora, Descriere, Reluare); missing parts are ''."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("div.descDetaliiObiecte h3")
    paragraphs = [p.get_text(strip=True) for p in soup.select("div.descDetaliiObiecte p")]

    def paragraph(i):
        return paragraphs[i] if i < len(paragraphs) else ""

    return {
        "Titlu": title.get_text(strip=True) if title else "",
        "Data": paragraph(P_DATE),
        "Ora": paragraph(P_TIME),
        "Descriere": paragraph(P_DESCRIPTION),
        "Reluare": paragraph(P_RESUMPTION),
    }

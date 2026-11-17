from accidents.scraping import parse_article, parse_listing

ARTICLE = """
<html><body><div class="descDetaliiObiecte">
  <h3> JUDEȚUL SIBIU: TRAFIC ÎNGREUNAT PE AUTOSTRADA A1 </h3>
  <p>Data: 03 Iulie 2025</p>
  <p>Ora: 15:50</p>
  <p>Info trafic</p>
  <p>Centrul INFOTRAFIC din Inspectoratul General al Poliției Române informează că ...</p>
  <p>Se estimează reluarea traficului în condiții normale după ora 16.30.</p>
</div></body></html>
"""


def test_parse_article():
    art = parse_article(ARTICLE)
    assert art["Titlu"].startswith("JUDEȚUL SIBIU")
    assert art["Data"] == "Data: 03 Iulie 2025"
    assert art["Ora"] == "Ora: 15:50"
    assert art["Descriere"].startswith("Centrul INFOTRAFIC")
    assert art["Reluare"].endswith("ora 16.30.")


def test_parse_article_missing_paragraphs_does_not_crash():
    # the old scraper raised IndexError here
    art = parse_article('<div class="descDetaliiObiecte"><p>Data: 1 Mai 2024</p></div>')
    assert art == {"Titlu": "", "Data": "Data: 1 Mai 2024", "Ora": "", "Descriere": "", "Reluare": ""}


def test_parse_listing():
    html = '<h3><a href="https://x/1">a</a></h3><h3><a href="https://x/2">b</a></h3><h3><a>no link</a></h3>'
    assert parse_listing(html) == ["https://x/1", "https://x/2"]
    assert parse_listing("<html></html>") == []

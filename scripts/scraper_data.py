from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import csv
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# Setăm calea către chromedriver
path_to_webdriver = "C:/chromedriver/chromedriver.exe"
service = Service(path_to_webdriver)
driver = webdriver.Chrome(service=service)

# Deschidem fișierul CSV o singură dată, în modul scriere
with open(ROOT / "data" / "accidente.csv", "a", encoding="utf-8-sig", newline="") as doc: #TODO: de modificat in w
    writer = csv.writer(doc)
    
    # Scriem anteturile coloanelor în fișier
    # writer.writerow(["Nr.crt.", "Titlu", "Data", "Ora", "Descriere", "Reluare"])TODO: de eliminat comentariu
    crt = 0  # Contor pentru numărul curent de înregistrare
    page = 107  # Pagină curentă de pe site TODO: de modificat page = 1

    while True:
        # Construim URL-ul pentru pagina curentă
        url = f"https://politiaromana.ro/ro/info-trafic?page={page}&statusActiv=0&categorie=2&data="
        driver.get(url)
        print(f"Pagina {page} conectată")
        time.sleep(1)  # Așteptăm încărcarea completă a paginii

        # Parsăm codul HTML cu BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = soup.select("h3 a")  # Selectăm toate linkurile către articole

        # Dacă nu mai sunt linkuri pe pagină, încheiem căutarea
        if not links:
            print("Căutare finalizată — nu mai sunt pagini cu articole.")
            break

        # Pentru fiecare link de articol de pe pagină
        for link in links:
            href = link.get("href")  # Extragem linkul complet
            driver.get(href)  # Navigăm la pagina articolului
            time.sleep(0.8)

            # Parsăm HTML-ul paginii articolului
            element_soup = BeautifulSoup(driver.page_source, "html.parser")

            # Inițializăm variabilele cu valori goale
            titlu = ""
            data = ""
            ora = ""
            descriere = ""
            reluare = ""

            # Extragem titlul articolului (dacă există)
            try:
                titlu = element_soup.select_one("div.descDetaliiObiecte h3").text.strip()
            except:
                pass  # Dacă nu există titlu, lăsăm gol

            # Extragem paragrafele (data, ora, descriere, reluare)
            paragrafe = element_soup.select("div.descDetaliiObiecte p")
            if len(paragrafe) > 0:
                data = paragrafe[0].text.strip()
                ora = paragrafe[1].text.strip()
                descriere = paragrafe[3].text.strip()
                reluare = paragrafe[4].text.strip()

            # Incrementez contorul și scriu rândul în CSV
            crt += 1
            writer.writerow([crt, titlu, data, ora, descriere, reluare])
            print(f"Articol salvat: {titlu}")

        # Trecem la pagina următoare
        page += 1

# Închidem browserul la final
driver.quit()
print("Proces complet.")
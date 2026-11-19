"""Step 1 (optional): scrape accident reports from politiaromana.ro.

Examples:
    python scripts/scrape_data.py                       # all pages -> data/raw_reports.csv (overwrites)
    python scripts/scrape_data.py --start-page 1 --end-page 5 --output data/sample.csv
    python scripts/scrape_data.py --start-page 107 --append   # resume an interrupted run

Requires Chrome; Selenium (>= 4.6) downloads a matching driver automatically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import time

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from accidents.config import RAW_CSV
from accidents.scraping import BASE_URL, CSV_HEADER, parse_article, parse_listing


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-page", type=int, default=1)
    p.add_argument("--end-page", type=int, default=None, help="last page to scrape (default: until no articles)")
    p.add_argument("--output", type=Path, default=RAW_CSV)
    p.add_argument("--append", action="store_true", help="append to the output file instead of overwriting it")
    p.add_argument("--delay", type=float, default=1.0, help="seconds to wait between requests (be polite)")
    p.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for a page to load")
    p.add_argument("--headed", action="store_true", help="show the browser window")
    return p.parse_args()


def make_driver(headed):
    options = webdriver.ChromeOptions()
    if not headed:
        options.add_argument("--headless=new")
    return webdriver.Chrome(options=options)


def load(driver, url, css, timeout):
    """Open `url` and wait until `css` is present; returns the HTML or None."""
    try:
        driver.get(url)
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, css)))
        return driver.page_source
    except TimeoutException:
        print(f"  timeout: {url}")
    except WebDriverException as exc:
        print(f"  browser error on {url}: {exc.msg}")
    return None


def last_row_number(path):
    """Continue numbering when appending to an existing file."""
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        return int(rows[-1][0]) if len(rows) > 1 else 0
    except (FileNotFoundError, ValueError, IndexError):
        return 0


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    append = args.append and args.output.exists()
    crt = last_row_number(args.output) if append else 0

    driver = make_driver(args.headed)
    try:
        with open(args.output, "a" if append else "w", encoding="utf-8-sig", newline="") as doc:
            writer = csv.writer(doc)
            if not append:
                writer.writerow(CSV_HEADER)

            page = args.start_page
            while args.end_page is None or page <= args.end_page:
                html = load(driver, BASE_URL.format(page=page), "body", args.timeout)
                links = parse_listing(html) if html else []
                if not links:
                    print(f"Page {page}: no articles - done.")
                    break
                print(f"Page {page}: {len(links)} articles")

                for href in links:
                    time.sleep(args.delay)
                    article_html = load(driver, href, "div.descDetaliiObiecte", args.timeout)
                    if article_html is None:
                        continue
                    article = parse_article(article_html)
                    crt += 1
                    writer.writerow([crt] + [article[c] for c in CSV_HEADER[1:]])
                    print(f"  saved: {article['Titlu'][:80]}")
                doc.flush()
                page += 1
                time.sleep(args.delay)
    finally:
        driver.quit()
    print(f"Done: {crt} rows in {args.output}")


if __name__ == "__main__":
    main()

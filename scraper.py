"""
Sri Lanka Property Price Scraper
Scrapes property listings from ikman.lk (houses and apartments for sale).
Two-phase approach:
  Phase 1: Collect ad slugs from listing pages
  Phase 2: Fetch detailed property data from individual ad pages
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import pandas as pd
import time
import random
import logging
from pathlib import Path
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_URL = "https://ikman.lk"
CATEGORIES = {
    "houses": "/en/ads/sri-lanka/houses-for-sale?page={page}",
}
MAX_LISTING_PAGES = 48  # ~1,200 listings (25 per page)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
REQUEST_TIMEOUT = 30
MIN_DELAY = 1.5
MAX_DELAY = 4.0
MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N pages / ads

OUTPUT_DIR = Path("data")
LISTINGS_CHECKPOINT = OUTPUT_DIR / "listings_checkpoint.csv"
DETAILS_CHECKPOINT = OUTPUT_DIR / "details_checkpoint.csv"
RAW_CSV = OUTPUT_DIR / "raw_data.csv"

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Helper Functions ────────────────────────────────────────────────────────


def extract_initial_data(html: str) -> dict | None:
    """Extract the window.initialData JSON from the page HTML."""
    # Find the start of the JSON assignment
    match = re.search(r"window\.initialData\s*=\s*", html)
    if not match:
        return None
    start = match.end()
    # Find the closing </script> tag after the JSON blob
    rest = html[start:]
    end = rest.find("</script>")
    if end == -1:
        return None
    json_str = rest[:end].rstrip().rstrip(";")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse initialData JSON: {e}")
        return None


def fetch_page(url: str) -> str | None:
    """Fetch a URL with retries and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url, headers=HEADERS, timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                wait = (2 ** attempt) * 5 + random.uniform(1, 3)
                logger.warning(f"Rate limited (429). Waiting {wait:.1f}s...")
                time.sleep(wait)
            elif response.status_code == 403:
                wait = (2 ** attempt) * 10 + random.uniform(2, 5)
                logger.warning(f"Forbidden (403). Waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                logger.warning(
                    f"HTTP {response.status_code} for {url}"
                )
                return None
        except requests.RequestException as e:
            wait = (2 ** attempt) * 3 + random.uniform(1, 2)
            logger.warning(f"Request error: {e}. Retry in {wait:.1f}s...")
            time.sleep(wait)
    logger.error(f"Failed after {MAX_RETRIES} retries: {url}")
    return None


def random_delay():
    """Sleep for a random duration to be polite."""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


# ─── Phase 1: Scrape Listing Pages ──────────────────────────────────────────


def parse_listing_page(data: dict, category: str) -> list[dict]:
    """Extract ad summaries from a listing page's initialData."""
    ads = []
    try:
        ads_data = data["serp"]["ads"]["data"]["ads"]
    except (KeyError, TypeError):
        logger.warning("Could not find ads in initialData")
        return ads

    for ad in ads_data:
        try:
            ads.append({
                "id": ad.get("id", ""),
                "slug": ad.get("slug", ""),
                "title": ad.get("title", ""),
                "price_str": ad.get("price", ""),
                "location_listing": ad.get("location", ""),
                "details_str": ad.get("details", ""),
                "category_name": category,
            })
        except Exception as e:
            logger.warning(f"Error parsing ad: {e}")
    return ads


def scrape_all_listings(max_pages_per_category: int = None) -> pd.DataFrame:
    """Phase 1: Scrape listing pages to collect all ad slugs."""
    # Resume from checkpoint if exists
    if LISTINGS_CHECKPOINT.exists():
        existing = pd.read_csv(LISTINGS_CHECKPOINT)
        logger.info(f"Resuming from checkpoint: {len(existing)} listings found")
    else:
        existing = pd.DataFrame()

    all_ads = []
    existing_slugs = set(existing["slug"].tolist()) if not existing.empty else set()

    for category, url_template in CATEGORIES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping {category} listings...")
        logger.info(f"{'='*60}")

        # Fetch first page to get total count
        first_url = BASE_URL + url_template.format(page=1)
        html = fetch_page(first_url)
        if not html:
            logger.error(f"Could not fetch first page for {category}")
            continue

        data = extract_initial_data(html)
        if not data:
            logger.error(f"No initialData found for {category}")
            continue

        # Get total pages
        try:
            pag = data["serp"]["ads"]["data"]["paginationData"]
            total_ads = pag.get("total", pag.get("nrAds", 0))
            page_size = pag.get("pageSize", 25)
            total_pages = (total_ads + page_size - 1) // page_size
        except (KeyError, TypeError):
            total_pages = 100  # fallback
            total_ads = "unknown"

        if max_pages_per_category:
            total_pages = min(total_pages, max_pages_per_category)

        scraping_count = total_pages * page_size
        logger.info(f"Scraping {category}: {scraping_count} ads ({total_pages} pages)")

        # Parse first page
        ads = parse_listing_page(data, category)
        new_ads = [a for a in ads if a["slug"] not in existing_slugs]
        all_ads.extend(new_ads)
        for a in new_ads:
            existing_slugs.add(a["slug"])

        # Scrape remaining pages
        for page in tqdm(range(2, total_pages + 1), desc=f"Listing pages ({category})"):
            url = BASE_URL + url_template.format(page=page)
            html = fetch_page(url)
            if not html:
                continue

            data = extract_initial_data(html)
            if not data:
                continue

            ads = parse_listing_page(data, category)
            new_ads = [a for a in ads if a["slug"] not in existing_slugs]
            all_ads.extend(new_ads)
            for a in new_ads:
                existing_slugs.add(a["slug"])

            # Checkpoint save
            if page % CHECKPOINT_INTERVAL == 0:
                combined = pd.concat(
                    [existing, pd.DataFrame(all_ads)], ignore_index=True
                )
                combined.to_csv(LISTINGS_CHECKPOINT, index=False)
                logger.info(f"Checkpoint: {len(combined)} total listings saved")

            random_delay()

    # Final save
    result = pd.concat(
        [existing, pd.DataFrame(all_ads)], ignore_index=True
    )
    result.drop_duplicates(subset="slug", keep="first", inplace=True)
    result.to_csv(LISTINGS_CHECKPOINT, index=False)
    logger.info(f"Phase 1 complete: {len(result)} total listings collected")
    return result


# ─── Phase 2: Scrape Detail Pages ───────────────────────────────────────────


def parse_detail_page(data: dict) -> dict:
    """Extract structured property data from a detail page's initialData."""
    result = {}

    # Navigate to the actual ad object: data -> adDetail -> data -> ad
    ad_data = {}
    try:
        ad_detail = data.get("adDetail", {})
        if isinstance(ad_detail, dict) and "data" in ad_detail:
            inner = ad_detail["data"]
            if isinstance(inner, dict) and "ad" in inner:
                ad_data = inner["ad"]
            elif isinstance(inner, dict):
                ad_data = inner
        if not ad_data:
            # Fallback: try direct paths
            ad_data = data.get("ad", {})
    except Exception:
        ad_data = {}

    # Price
    try:
        money = ad_data.get("money", {})
        result["price_detail"] = money.get("amount", "")
    except Exception:
        result["price_detail"] = ""

    # Location
    try:
        location = ad_data.get("location", {})
        result["city"] = location.get("name", "")
        result["city_slug"] = location.get("slug", "")
        parent = location.get("parent", {})
        result["district"] = parent.get("name", "")
        result["district_slug"] = parent.get("slug", "")
    except Exception:
        result["city"] = ""
        result["district"] = ""
        result["city_slug"] = ""
        result["district_slug"] = ""

    # Properties (bedrooms, bathrooms, house_size, land_size, address)
    try:
        properties = ad_data.get("properties", [])
        for prop in properties:
            key = prop.get("key", "")
            value = prop.get("value", "")
            if key in ("bedrooms", "bathrooms", "house_size", "land_size", "address"):
                result[key] = value
    except Exception:
        pass

    # Ensure all property fields exist
    for field in ("bedrooms", "bathrooms", "house_size", "land_size", "address"):
        if field not in result:
            result[field] = ""

    # Description
    result["description"] = ad_data.get("description", "")

    # Category
    try:
        result["category_detail"] = ad_data.get("category", {}).get("name", "")
    except Exception:
        result["category_detail"] = ""

    return result


def scrape_all_details(listings_df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: Fetch detail pages for structured property data."""
    # Resume from checkpoint if exists
    if DETAILS_CHECKPOINT.exists():
        existing = pd.read_csv(DETAILS_CHECKPOINT)
        scraped_slugs = set(existing["slug"].tolist())
        logger.info(f"Resuming details: {len(existing)} already scraped")
    else:
        existing = pd.DataFrame()
        scraped_slugs = set()

    # Filter to unscraped slugs
    remaining = listings_df[~listings_df["slug"].isin(scraped_slugs)]
    logger.info(f"Detail pages to scrape: {len(remaining)}")

    new_details = []

    for idx, row in tqdm(
        remaining.iterrows(), total=len(remaining), desc="Detail pages"
    ):
        slug = row["slug"]
        url = f"{BASE_URL}/en/ad/{slug}"

        html = fetch_page(url)
        if not html:
            new_details.append({"slug": slug, "scrape_error": "fetch_failed"})
            random_delay()
            continue

        data = extract_initial_data(html)
        if not data:
            new_details.append({"slug": slug, "scrape_error": "no_initial_data"})
            random_delay()
            continue

        detail = parse_detail_page(data)
        detail["slug"] = slug
        detail["scrape_error"] = ""
        new_details.append(detail)

        # Checkpoint save
        if len(new_details) % CHECKPOINT_INTERVAL == 0:
            new_df = pd.DataFrame(new_details)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(DETAILS_CHECKPOINT, index=False)
            logger.info(f"Checkpoint: {len(combined)} details saved")

        random_delay()

    # Final save
    new_df = pd.DataFrame(new_details)
    result = pd.concat([existing, new_df], ignore_index=True)
    result.to_csv(DETAILS_CHECKPOINT, index=False)
    logger.info(f"Phase 2 complete: {len(result)} detail records")
    return result


# ─── Merge & Clean Raw Data ─────────────────────────────────────────────────


def parse_price(price_str: str) -> float | None:
    """Convert price string like 'Rs 36,000,000' to float."""
    if not price_str or not isinstance(price_str, str):
        return None
    # Remove currency and commas
    cleaned = re.sub(r"[^\d.]", "", price_str.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def merge_data(listings_df: pd.DataFrame, details_df: pd.DataFrame) -> pd.DataFrame:
    """Merge listing and detail data into final raw dataset."""
    merged = listings_df.merge(details_df, on="slug", how="inner")

    # Parse price - prefer detail page price, fallback to listing price
    merged["price"] = merged["price_detail"].apply(parse_price)
    mask = merged["price"].isna()
    merged.loc[mask, "price"] = merged.loc[mask, "price_str"].apply(parse_price)

    # Select and rename columns
    final = merged[[
        "slug", "title", "price", "price_str",
        "city", "district", "address",
        "bedrooms", "bathrooms", "house_size", "land_size",
        "category_name", "description",
    ]].copy()

    final.rename(columns={"category_name": "property_type"}, inplace=True)

    # Drop rows without price
    initial_len = len(final)
    final = final.dropna(subset=["price"])
    final = final[final["price"] > 0]
    logger.info(f"After price filter: {len(final)}/{initial_len} rows")

    final.to_csv(RAW_CSV, index=False)
    logger.info(f"Raw data saved: {RAW_CSV} ({len(final)} rows)")
    return final


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    """Run the full scraping pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Sri Lanka Property Scraper - Starting")
    logger.info("=" * 60)

    # Phase 1: Collect listings
    logger.info("\nPHASE 1: Scraping listing pages...")
    listings = scrape_all_listings(max_pages_per_category=MAX_LISTING_PAGES)
    logger.info(f"Total listings collected: {len(listings)}")

    # Phase 2: Fetch details
    if listings.empty or "slug" not in listings.columns:
        logger.error("No listings collected in Phase 1. Aborting.")
        return
    logger.info("\nPHASE 2: Scraping detail pages...")
    details = scrape_all_details(listings)

    # Merge
    logger.info("\nMerging listing and detail data...")
    raw_data = merge_data(listings, details)

    logger.info("\n" + "=" * 60)
    logger.info(f"SCRAPING COMPLETE: {len(raw_data)} properties saved to {RAW_CSV}")
    logger.info("=" * 60)

    # Print summary
    print("\n--- Dataset Summary ---")
    print(f"Total records: {len(raw_data)}")
    print(f"\nProperty types:\n{raw_data['property_type'].value_counts()}")
    print(f"\nDistricts:\n{raw_data['district'].value_counts().head(10)}")
    print(f"\nPrice range: Rs {raw_data['price'].min():,.0f} - Rs {raw_data['price'].max():,.0f}")
    print(f"Median price: Rs {raw_data['price'].median():,.0f}")
    print(f"\nMissing values:\n{raw_data.isnull().sum()}")


if __name__ == "__main__":
    main()

# scraping/sports/football_scraper.py

from app.services.scraping_service import scrape_data_and_store

def scrape_football_data():
    """Scrape football data and save it to MongoDB."""
    print("Starting football data scraping...")
    scrape_data_and_store('football')



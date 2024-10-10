from app.services.scraping_service import scrape_data_and_store

def scrape_cricket_data():
    """Scrape cricket data and save it to MongoDB."""
    print("Starting cricket data scraping...")
    scrape_data_and_store('cricket')

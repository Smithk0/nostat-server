from app.services.scraping_service import scrape_data_and_store

def scrape_handball_data():
    """Scrape handball data and save it to MongoDB."""
    print("Starting handball data scraping...")
    scrape_data_and_store('handball')

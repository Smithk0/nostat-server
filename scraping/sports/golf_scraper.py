from app.services.scraping_service import scrape_data_and_store

def scrape_golf_data():
    """Scrape golf data and save it to MongoDB."""
    print("Starting golf data scraping...")
    scrape_data_and_store('golf')

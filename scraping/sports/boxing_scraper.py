from app.services.scraping_service import scrape_data_and_store

def scrape_boxing_data():
    """Scrape boxing data and save it to MongoDB."""
    print("Starting boxing data scraping...")
    scrape_data_and_store('boxing')

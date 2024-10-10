from app.services.scraping_service import scrape_data_and_store

def scrape_soccer_data():
    """Scrape soccer data and save it to MongoDB."""
    print("Starting soccer data scraping...")
    scrape_data_and_store('soccer')

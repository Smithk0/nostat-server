from app.services.scraping_service import scrape_data_and_store

def scrape_volleyball_data():
    """Scrape volleyball data and save it to MongoDB."""
    print("Starting volleyball data scraping...")
    scrape_data_and_store('volleyball')

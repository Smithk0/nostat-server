from app.services.scraping_service import scrape_data_and_store

def scrape_baseball_data():
    """Scrape baseball data and save it to MongoDB."""
    print("Starting baseball data scraping...")
    scrape_data_and_store('baseball')

from app.services.scraping_service import scrape_data_and_store

def scrape_beach_volleyball_data():
    """Scrape beach volleyball data and save it to MongoDB."""
    print("Starting beach volleyball data scraping...")
    scrape_data_and_store('beach_volleyball')

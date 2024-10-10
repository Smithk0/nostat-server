from app.services.scraping_service import scrape_data_and_store

def scrape_basketball_data():
    """Scrape basketball data and save it to MongoDB."""
    print("Starting basketball data scraping...")
    scrape_data_and_store('basketball')

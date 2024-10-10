from app.services.scraping_service import scrape_data_and_store

def scrape_tennis_data():
    """Scrape tennis data and save it to MongoDB."""
    print("Starting tennis data scraping...")
    scrape_data_and_store('tennis')

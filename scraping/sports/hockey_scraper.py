from app.services.scraping_service import scrape_data_and_store

def scrape_hockey_data():
    """Scrape hockey data and save it to MongoDB."""
    print("Starting hockey data scraping...")
    scrape_data_and_store('hockey')

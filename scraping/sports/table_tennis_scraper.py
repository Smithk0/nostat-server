from app.services.scraping_service import scrape_data_and_store

def scrape_table_tennis_data():
    """Scrape table tennis data and save it to MongoDB."""
    print("Starting table tennis data scraping...")
    scrape_data_and_store('table_tennis')

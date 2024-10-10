from app.services.scraping_service import scrape_data_and_store

def scrape_snooker_data():
    """Scrape snooker data and save it to MongoDB."""
    print("Starting snooker data scraping...")
    scrape_data_and_store('snooker')

from app.services.scraping_service import scrape_data_and_store

def scrape_badminton_data():
    """Scrape badminton data and save it to MongoDB."""
    print("Starting badminton data scraping...")
    scrape_data_and_store('badminton')

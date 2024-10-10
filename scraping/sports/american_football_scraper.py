from app.services.scraping_service import scrape_data_and_store

def scrape_american_football_data():
    """Scrape American football data and save it to MongoDB."""
    print("Starting American football data scraping...")
    scrape_data_and_store('american_football')

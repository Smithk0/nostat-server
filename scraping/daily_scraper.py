# scraping/daily_scraper.py

import sys
import os
import asyncio  # Import asyncio to run the async function

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from app.services.scraping_service import scrape_daily_results_and_stats

async def daily_scrape():
    """Scrape today's football results and statistics, and store them in MongoDB."""
    await scrape_daily_results_and_stats()

if __name__ == '__main__':
    asyncio.run(daily_scrape())
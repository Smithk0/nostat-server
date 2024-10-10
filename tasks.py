# tasks.py

import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.services.scraping_service import scrape_daily_results_and_stats
from app.services.service_update import monitor_events, update_all_prediction_outcomes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_daily_scraper():
    """
    Run the daily_scraper.py functionality.
    """
    logger.info("Starting daily scraper task...")
    try:
        await scrape_daily_results_and_stats()
        logger.info("Daily scraper task completed successfully.")
    except Exception as e:
        logger.error(f"Daily scraper task failed: {e}")

async def run_service_update():
    """
    Run the service_update.py functionality.
    """
    logger.info("Starting service update task...")
    try:
        await monitor_events()
        await update_all_prediction_outcomes()
        logger.info("Service update task completed successfully.")
    except Exception as e:
        logger.error(f"Service update task failed: {e}")

def main():
    """
    Main entry point for scheduling tasks.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    # Schedule daily_scraper to run at 12:00 AM UTC and 6:00 PM UTC
    daily_scraper_trigger = CronTrigger(hour='0,18', minute='0')
    scheduler.add_job(run_daily_scraper, trigger=daily_scraper_trigger, id='daily_scraper')

    # Schedule service_update to run every 20 minutes
    service_update_trigger = IntervalTrigger(minutes=20)
    scheduler.add_job(run_service_update, trigger=service_update_trigger, id='service_update')

    scheduler.start()
    logger.info("Scheduler started. Press Ctrl+C to exit.")

    try:
        # Keep the script running to allow scheduler to run jobs
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()

if __name__ == "__main__":
    main()

import logging
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

logger = logging.getLogger('django')

def test():
    logger.info('test cron')
import logging
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

logger = logging.getLogger('django')
import tensorflow as tf

logger.info('test cron')
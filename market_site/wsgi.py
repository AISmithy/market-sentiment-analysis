import os

"""\
© 2025 Nishant Kumar. Confidential and Proprietary.
Unauthorized copying, distribution, modification, or use of this software
is strictly prohibited without express written permission.
"""

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'market_site.settings')

application = get_wsgi_application()

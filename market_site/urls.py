"""\
© 2025 Nishant Kumar. Confidential and Proprietary.
Unauthorized copying, distribution, modification, or use of this software
is strictly prohibited without express written permission.
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('sentiment.urls')),
]

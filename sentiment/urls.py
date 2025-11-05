"""\
© 2025 Nishant Kumar. Confidential and Proprietary.
Unauthorized copying, distribution, modification, or use of this software
is strictly prohibited without express written permission.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze, name='analyze'),
    path('price/', views.price, name='price'),
    path('history/', views.history, name='history'),
]

"""
© 2025 Nishant Kumar. Confidential and Proprietary.
Unauthorized copying, distribution, modification, or use of this software
is strictly prohibited without express written permission.
"""

from django.urls import path
from . import views
from . import predict_views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze, name='analyze'),
    path('price/', views.price, name='price'),
    path('history/', views.history, name='history'),
    path('predict/', predict_views.predict, name='predict'),
]

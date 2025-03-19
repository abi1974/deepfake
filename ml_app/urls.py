"""project_settings URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import about, index, predict_page,cuda_full,home,load

app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
    
    path('index/', index, name='index'),
    path('about/', about, name='about'),
    path('predict/', predict_page, name='predict'),
    path('cuda_full/',cuda_full,name='cuda_full'),
   path('home/', home, name='home'),
    path('',load,name='load'),
     
   
  
  
]



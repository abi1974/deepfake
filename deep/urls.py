from . import views
from django.urls import path
from .views import upload,result,invalid_image

urlpatterns=[

       path('upload/',upload,name="upload"),
    path('results/',result,name="result"),
    path('invalid_image/',invalid_image,name="invalid_image"),
]
app_name = "deep" 
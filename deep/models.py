
from django.db import models

# Create your models here.
from django.db import models

class Image(models.Model):
    image = models.ImageField(upload_to="img/%y")

    def __str__(self):
        return self.image.name  # You can return any meaningful representation here

# Create your models here.

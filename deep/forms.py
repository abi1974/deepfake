
from django import forms

from .models import Image

class ImageForm(forms.ModelForm):
    class Meta:
        app_label = 'deep'
        model=Image
        fields=("image",)
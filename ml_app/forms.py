from django import forms

class VideoUploadForm(forms.Form):

    upload_video_file = forms.FileField(label="Select Video", required=True,widget=forms.FileInput(attrs={"accept": "video/*"}))
    sequence_length = forms.IntegerField(label="Sequence Length", required=True)



from .models import Image

class ImageForm(forms.ModelForm):
    class Meta:
        app_label = 'deep'
        model=Image
        fields=("caption","image")




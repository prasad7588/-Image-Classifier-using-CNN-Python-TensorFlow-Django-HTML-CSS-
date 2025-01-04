# image_classifier/forms.py
from django import forms

class ImageForm(forms.Form):
    image = forms.ImageField()  # Image input field

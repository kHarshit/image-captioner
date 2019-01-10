from django import forms
from django.core.validators import FileExtensionValidator


class FileForm(forms.Form):
    """Form handling uploaded image"""
    file = forms.ImageField(validators=[FileExtensionValidator(['png', 'jpg', 'jpeg'])])

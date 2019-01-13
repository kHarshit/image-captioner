from django import forms
from .models import File


class FileForm(forms.ModelForm):
    """Form handling uploaded image"""
    class Meta:
        model = File
        fields = ('file',)

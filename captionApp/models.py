from django.db import models
from django.core.validators import FileExtensionValidator


class File(models.Model):
    """model for files
    :file: image file
    :datetime: date-time info when file uploaded"""
    file = models.ImageField(validators=[FileExtensionValidator(['png', 'jpg', 'jpeg'])])
    uploaded_at = models.DateTimeField(auto_now_add=True)

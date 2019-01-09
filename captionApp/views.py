from django.shortcuts import render


def index(request):
    """homepage"""
    return render(request, template_name='captionApp/index.html')

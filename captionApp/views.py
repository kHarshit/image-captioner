from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
# from pytorch.prediction import process
# import PIL.Image

from .forms import FileForm


# def index(request):
#     """homepage"""
#     return render(request, template_name='captionApp/index.html')


def handle_file(image):
    """process uploaded file"""
    # img = PIL.Image.open(image)
    # print(img.size)
    pass


def index(request):
    """homepage"""
    if request.method == 'POST':
        form = FileForm(request.POST, request.FILES)
        if form.is_valid():
            myfile = request.FILES['file']
            print(myfile)
            handle_file(myfile)
            return render(request, 'captionApp/output.html', {'file': myfile})
            # return HttpResponse('File successfully uploaded!')
        # else:
        #     return HttpResponse('Something Went Wrong!')
    else:
        form = FileForm()
    return render(request, 'captionApp/index.html', {'form': form})



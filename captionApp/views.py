from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
# from pytorch.prediction import process
# import PIL.Image
import copy

from .forms import FileForm
from .models import File


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
            myfile = File(file=request.FILES['file'])
            form.save()
            file_name = myfile.file  # file attribute of File model object
            print(file_name)
            file_name_original = copy.copy(file_name)  # prevent name modification after saving file
            handle_file(myfile)
            myfile.save()
            print(file_name)
            return render(request, 'captionApp/output.html', {'file_name': file_name,
                                                              'file_name_original': file_name_original})
            # return HttpResponse('File successfully uploaded!')
        # else:
        #     return HttpResponse('Something Went Wrong!')
    else:
        form = FileForm()
    return render(request, 'captionApp/index.html', {'form': form})



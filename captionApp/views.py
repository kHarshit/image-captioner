from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
import copy
from .forms import FileForm
from .models import File
from .prediction import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Vocabulary':
            return Vocabulary
        return super().find_class(module, name)


vocab = CustomUnpickler(open('/home/harshit/Downloads/vocab.pkl', 'rb')).load()


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
            # process image
            output = process(file_name,
                             '/home/harshit/Downloads/pretrained_model/encoder-5-3000.pkl',
                             '/home/harshit/Downloads/pretrained_model/decoder-5-3000.pkl',
                             vocab,
                             256, 512, 1)
            print(output)
            myfile.save()
            print(file_name)
            return render(request, 'captionApp/output.html', {'file_name': file_name,
                                                              'file_name_original': file_name_original,
                                                              'output': output})
            # return HttpResponse('File successfully uploaded!')
        # else:
        #     return HttpResponse('Something Went Wrong!')
    else:
        form = FileForm()
    return render(request, 'captionApp/index.html', {'form': form})


### For heroku  ###
# def index(request):
#     """homepage"""
#     if request.method == 'POST':
#         form = FileForm(request.POST, request.FILES)
#         if form.is_valid():
#             myfile = request.FILES['file']
#             print(myfile)
#             handle_file(myfile)
#             return render(request, 'captionApp/output.html', {'file': myfile})
#             # return HttpResponse('File successfully uploaded!')
#         # else:
#         #     return HttpResponse('Something Went Wrong!')
#     else:
#         form = FileForm()
#     return render(request, 'captionApp/index.html', {'form': form})

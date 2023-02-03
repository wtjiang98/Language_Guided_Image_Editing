from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm
from django.core.files import File

import pexpect
import os.path
import time
import base64
import subprocess
import json


def home(request):
    documents = Document.objects.all()
    for doc in documents:
        doc.inference = "/media/" + "inference_"+doc.document.url.split("/")[-1]
    return render(request, 'core/home.html', { 'documents': documents })


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'core/simple_upload.html')


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            if os.path.exists("media"):
                print("exists")
                pass
            form.save()
            myfile = request.FILES['document']
            query = request.POST['request']
            sequenece = False if len(request.POST) == 2 else True
            print(sequenece)
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            
            filename_base64 = subprocess.check_output("echo "+filename+" | base64 ", shell=True)
            filename_base64 = filename_base64[0:8].decode()
            print("name_base64 is ", filename_base64 )
            
            # print(fs.url())
            if sequenece:
                '''
                 *******************************************
                 change the command in the following line
                 *******************************************   
                '''
                command = "bash run_remote_multi.sh media/" +str(filename)+ " \"" +str(query)+"\" media/"
                os.system(command)
                print(command)
                prefix = 1
                while fs.exists(str(prefix)+"_inference_" + str(filename_base64) + ".jpg"):
                    prefix+=1
                print("prefix: ",prefix)

                
                with open("media/"+str(filename_base64)+'.json', 'r') as f:
                    distros_dict = json.load(f)

                operations = distros_dict[0]['operations']

                output_file_url_list = []
                for index in range(1,prefix):
                    name = operations[index-1][0]
                    if name == "color" or name == "tone":
                        arg = ""
                    else:
                        arg =   "{:.7f}".format(operations[index-1][1][0]) 
                    url = fs.url(str(index)+"_inference_"+str(filename_base64)+".jpg")
                    op = str(name)+ " " + arg
                    print(op,url)
                    output_file_url_list.append((op,url))
                    # output_file_url_list.append(fs.url(str(index)+"_inference_"+str(filename_base64)+".jpg"))
                print("Here are the results:")
                print(output_file_url_list)

                return render(request, 'core/model_form_upload.html', {
                    'query':query,
                    'show_sequenece':sequenece,
                    'uploaded_file_url': uploaded_file_url,
                    'output_file_url_list':output_file_url_list
                })

            # command_FiveK = f"expect run_remote_single_jwt.sh ../media/{filename}  \"{query}\" ../media/inference_{filename}"
            command_GIER = f"expect run_remote_single_GIER_jwt.sh ../media/{filename}  \"{query}\" ../media/inference_{filename}"

            print(command_GIER)
            os.system(command_GIER)

            output_file_url = fs.url("inference_"+str(filename))
            output_file_url = output_file_url.replace('jpg', 'png')
            # print(output_file_url)
            return render(request, 'core/model_form_upload.html', {
                'query':query,
                'show_sequenece':sequenece,
                'uploaded_file_url': uploaded_file_url,
                'output_file_url':output_file_url
            })
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })


from posixpath import dirname
from flask  import Flask, request, render_template,redirect, url_for,abort,send_from_directory
from werkzeug.utils import secure_filename
import os.path
import tempfile
import io
import os
import base64
from datetime import datetime
from pathlib import Path

import torchvision
from torchvision import  transforms 
import torch
from torch import no_grad

import cv2
import numpy as np
from PIL import Image



# Here are the 91 classes.
OBJECTS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# Here are the  classesj for display 
OBJECTS_html=['all', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',  'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe',  'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',  'dining table', 'toilet',  'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#key type of objects  valuue:  list of files that pertain to each objects 
FILE_OBJ={}



def get_predictions(pred,threshold=0.8,objects=None ):
    """
    This function will assign a string name to a predicted class and eliminate predictions whose likelihood  is under a threshold 
    
    pred: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class yhat, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    thre
    """


    predicted_classes= [(OBJECTS[i],p,[(box[0], box[1]), (box[2], box[3])]) for i,p,box in zip(list(pred[0]['labels'].numpy()),pred[0]['scores'].detach().numpy(),list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes=[  stuff  for stuff in predicted_classes  if stuff[1]>threshold ]
    
    if objects  and predicted_classes :
        predicted_classes=[ (name, p, box) for name, p, box in predicted_classes if name in  objects ]
    return predicted_classes

def draw_box(predicted_classes,image,rect_th= 30,text_size= 3,text_th=3):
    """
    draws box around each object 
    
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface 
   
    """

    img=(np.clip(cv2.cvtColor(np.clip(image.numpy().transpose((1, 2, 0)),0,1), cv2.COLOR_RGB2BGR),0,1)*255).astype(np.uint8).copy()
    for predicted_class in predicted_classes:
   
        label=str(predicted_class[0]) + " likelihood"
        probability=predicted_class[1]
        box=predicted_class[2]
        cv2.rectangle(img, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])),(0, 255, 0), 4) # Draw Rectangle with the coordinates
        cv2.putText(img,label, (int(box[0][0]), int(box[0][1])),  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=3) 
        cv2.putText(img,label+": "+str(round(probability,2)), (int(box[0][0]), int(box[0][1])),  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=3)
    
    return img


#Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image  pre-trained on COCO.
model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# set to eval 
model_.eval()
# save memory
for name, param in model_.named_parameters():
    param.requires_grad = False
#the function calls Faster R-CNN  model_  but save RAM:
def model(x):
    with torch.no_grad():
        yhat = model_(x)
    return yhat 
#  transform image to tensor 
transform = transforms.Compose([transforms.ToTensor()])


app=Flask(__name__)

# EXTENSIONS allowed  
dostuff=None
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif','.jpeg']

# paths 
app.config['UPLOAD_PATH'] = 'uploads'
app.config['OBJECTS_PATH'] = 'objects' 

# confident_range
app.config['CONFIDENT_RANG'] = None

#path of  images 
app.config['FILE_PATH']=None
app.config['FILE_NAME']=[]
# directory of path 
dir_name = Path(app.instance_path)
@app.route('/')
def home():
    #new file that  has  been uploaded 
   
    files= os.listdir(app.config['UPLOAD_PATH'])
    
    # check  if  a the following 
    files=[ file  for file in files  if os.path.splitext(file )[1] in app.config['UPLOAD_EXTENSIONS'] ]
    #files that  has  been uploaded that have been  uploaded 
    object_files=os.listdir(app.config['OBJECTS_PATH'])
    object_files=[ file  for file in object_files  if os.path.splitext(file )[1] in app.config['UPLOAD_EXTENSIONS'] ]
    return render_template('index.html', files=app.config['FILE_NAME'] ,objects_list=OBJECTS_html,object_files=object_files)



@app.route('/', methods=['POST'])
def upload_file():
    #file object
    uploaded_file = request.files['file']
    #file name  
    filename= secure_filename(uploaded_file.filename)
    #file extention 
    file_ext = os.path.splitext(filename)[1]

    #check if empty file 
    if filename != '':
        # file path /uploads/filename 

        #check if .jpg, .png, .gif if  not send an error 
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        #send back  to home  agument is the fuction "home"
        #upload file  path 
        #uploaded_file.save(filename)
        file_path=Path(app.config['UPLOAD_PATH']).joinpath(filename)
        # same the file name to be used in other parts of app  
        app.config['FILE_NAME']=[filename]
        # file path to be used in app
        app.config['FILE_PATH']=file_path

        uploaded_file.save(file_path)
        return redirect(url_for('home'))

@app.route('/find_object', methods=['POST']) 
def find():
    redirect(url_for('home'))
    #  object  to find 
    object=request.form.get("objects")
    confident_range = request.form.get("confident_range")
    app.config['CONFIDENT_RANG'] = int(confident_range) / int(100)
    print("++++++++", confident_range)
    # this is a bug fix as it will only save the image twice
    object_=object
    if object_:
        half = 0.5
        print(app.config['FILE_PATH'])
        image = Image.open(app.config['FILE_PATH'])
        
        arr = []
        image.resize( [int(half * s) for s in image.size] ) 
        img = transform(image)
        pred = model(torch.unsqueeze(img,0))   

        if object=='all':
            pred_thresh=get_predictions(pred,threshold=app.config['CONFIDENT_RANG'])
        

        else:
            pred_thresh=get_predictions(pred,threshold=app.config['CONFIDENT_RANG'],objects=object)
       
        object_=None   
        #draw box on image 
        image_numpy=draw_box(pred_thresh,img,rect_th= 1,text_size= 1,text_th=1) 
        #save image with box with new name 
        filename, file_extension = os.path.splitext(app.config['FILE_NAME'][0])
        print(filename, file_extension)
        app.config['FILE_NAME']  = []
        #name of file with lables 
        new_file_name=filename+"_object"+file_extension
        new_file_path=Path(app.config['OBJECTS_PATH']).joinpath(new_file_name)

        #save file we  use opencv as the  boxes look better 
        cv2.imwrite(str(new_file_path), image_numpy)

        #get differnet objects and save as image 

        for obj in pred_thresh:
  
            #Top and bottom corner of box 
            x_0,y_0=obj[2][0]
            x_1,y_1=obj[2][1]
            #save the image with a name that inculds the object and time

            now = datetime.now()
            dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S_%f").strip()
            obj_file_name=obj[0]+dt_string+file_extension
            object_file_ext=Path(app.config['OBJECTS_PATH']).joinpath(obj_file_name)
        
            if not(obj[0] in set(FILE_OBJ.keys())):
                FILE_OBJ[obj[0]]=[obj_file_name ]
            else:
                FILE_OBJ[obj[0]].append(obj_file_name)
           
            new_image=image.copy().crop((x_0,y_0,x_1,y_1))
    
            new_image.save(object_file_ext)

    if (request.form.get("Find_New")):
        os.remove(app.config['FILE_PATH'])
        return redirect(url_for('home'))
    
    return render_template("find_object.html" ,objects=object,file=new_file_name, title=object, range1=confident_range)  



@app.route('/your_object')    
def your_gallery(): 
        print('assss',FILE_OBJ)
        return render_template("your_object.html" ,obj_files=FILE_OBJ)     
 #serve these uploade files from  following route
@app.route('/uploads/<filename>')
def upload(filename):
    #get file this is called in index.html 
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

 #serve these  files from  following routey
@app.route('/objects/<filename>')
def upload_objects(filename):
    #get file this is called in index.html 
    return send_from_directory(app.config['OBJECTS_PATH'], filename)    
@app.route('/your_object/<galleryName>')
def view_obejct(galleryName):
    return render_template("view_obejct.html" ,obj_files=FILE_OBJ[galleryName], title=galleryName)  
@app.route('/your_galary')
def view_gallery():
    files = os.listdir(app.config['UPLOAD_PATH'])
    print("test")
    return render_template("your_galary.html" ,obj_files=files)  



if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)



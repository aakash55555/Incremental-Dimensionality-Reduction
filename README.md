# An Interactive Visual Analytics System for Incremental Classification Based on Semi-supervised Topic Modeling

This project is a re-implementation of the paper titled as "An Interactive Visual Analytics System for Incremental Classification Based on Semi-supervised Topic Modeling" with few meaningful extensions in the form of additional visualization techniques.

**Prerequisites:**
Make sure you have Python3 and following python-libraries installed on your machine for smooth execution of the project.
- Numpy
- Scipy

You can install them using following commands:
```
pip install numpy
pip install scipy
```

To get started with the project, first clone the repository by executing following command in shell or cmd.
```
git clone https://github.com/asu-cse578-f2019/Incremental-Dimensionality-Reduction-team2-Kaushal-Kalpana-Varun-Aakash-Anchit.git
```
Once cloned, go inside the project directory and start the server using following command:
```
python -m http.server                # for python 3
python -m SimpleHttpServer           # for python 2
```
Once server started open another shell or cmd go into the project directory and go inside the Scripts directoery and run following command:
```
python GibbsMEDLDA.py
```
Make sure you have python3 installed on your system.

Cool! The setup is ready and you are good to go! Open Mozilla firefox or Google Chrome and visit http://localhost:8000/?ds=1 to enjoy this project!

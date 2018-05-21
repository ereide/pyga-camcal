# Introduction

This package is made for the development of my master thesis at the University of Cambridge titled "Camera Calibration using Conformal Geometric Algebra". The thesis presented a new way of approaching rotor estimation problems and showed how this can be used to calibrate external camera parameters. 

This repository includes the code used to develop and test the thoery presented in my thesis. However, this repo is not actively maintained as I have submitted my project report. If you do intend to use the code in this repository I cannot guarantee that it is all working as intended. However, I still ask you to reference this repository in any worked derived from it. 


# Setup

Check that you have virtualenv installed (linux).

```
    $pip install virtualenv
    $pip install virtualenv --upgrade
```

Create a virtual enviroment with Python3.
Ubuntu: ```$virtualenv -p python3 venv```
Windows + Python >3.4: ```$python -m venv venv```

Activate the virtual enviroment with:
Ubuntu: ``` $source venv/bin/activate ```
Windows: ``` $./venv/Scripts/activate ```

Install the requirments 
```
    $pip install -r requirements.txt 
```

Install the module
```
    $python setup.py build
    $python setup.py install
```


Everything should be good to go. 

The pip installed version of Clifford is too slow for the purposes of optimazation,
we therefore recommend installing the development version of clifford. 

```
    $source venv/bin/activate

    $git submodule init 
    $git submodule update

    $cd extern/clifford
    $python setup.py build
    $python setup.py install
```


Building docs:
```
    $cd docs/
    $make html
```

## Note on performance 

The code in this project can be made orders of magnitude faster using the numerical optimazations from Hugo Hadifield (@hugohadfield) in his clifford tool library. However, this is not publically available at the moment (as of May 2018), but as soon as it is bottleneck functions can be replaced using functions with the exact same signature from his library. 

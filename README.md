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

The pip installed version of clifford is too slow for the purposes of optimazation,
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

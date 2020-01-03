# About
This is python API of depthai and examples.  

# Python modules
Files with extention `.so` are python modules:  
- `depthai.cpython-36m-x86_64-linux-gnu.so` built for Ubuntu 18.04 & Python 3.6  
- `depthai.cpython-37m-arm-linux-gnueabihf.so` built for Raspbian 10 & Python 3.7  
  
# Examples
`test.py` - depth example  
`test_cnn.py` - CNN inference example

# Calibration
For better device performance, you need to calibrate it. To do it, you have to:
1. Print the chessboard for calibration. The picture can be found in the `resources` folder (resources/calibration-chess-board.png)
2 Start python3 script: type `python3 calibration_pipeline.py` in the terminal. Two streams left and right will show up. Each window will contain a polygon.
3. Put a printed chessboard within the polygon and press barspace. It will take a photo. There will be 13 positions of polygons.
4. After it, the calibration will automatically start based on taken pictures. If calibration is a successful file named `depthai.calib` will be generated. 

Depthai has the default calibration file. There are two ways to change it:
1. Easy way: rename your calibration file to `default.calib` and move it the resources folder
2. Harder way: go to the `consts/resource_paths.py` and set the path to your calibration file into `calib_fpath` variable.

# Issues reporting  
We are developing depthai framework, and it's crucial for us to know what kind of problems users are facing.  
So thanks for testing DepthAI! The information you give us, the faster we will help you and make depthai better!  
  
Please, do the following steps:  
1. Run script `log_system_information.sh` and provide us the output (`log_system_information.txt`, it's system version & modules versions);  
2. Take a photo of a device you are using (or provide us a device model);  
3. Describe the expected results;  
4. Describe the actual running results (what you see after started your script with depthai);  
5. Provide us information about how you are using the depthai python API (code snippet, for example);  
6. Send us consol outputs;  


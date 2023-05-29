## Generation of DetectionBestCandidate.blob
1) pytorch is needed
2) Generate the ONNX model *DetectionBestCandidate.onnx* : 
```
python DetectionBestCandidate.py
``` 

3) Start the tflite2tensorflow docker container:
```
../docker_tflite2tensorflow.sh
```
4) From the container shell:
```
cd workdir
./convert_model.sh -m DetectionBestCandidate # Use `-n #` if you want to change the number of shaves
```

## Generation of DivideBy255.blob
Same as above with 'DivideBy255' instead of 'DetectionBestCandidate'.


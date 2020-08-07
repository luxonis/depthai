import consts.resource_paths
import cv2
import depthai

if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
    raise RuntimeError("Error initializing device. Try to reset it.")
    
p = depthai.create_pipeline(config={
    "streams": ["previewout"],
    "ai": {
        "blob_file": f"/home/one/depth_test/depthai-python-extras/resources/nn/mobilenet-ssd/openpose.blob",
        "blob_file_config": f"/home/one/depth_test/depthai-python-extras/resources/nn/mobilenet-ssd/openpose.json"
    }
})
if p is None:
    raise RuntimeError("Error initializing pipelne")
entries_prev = []

while True:
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()    
    for packet in data_packets:
        if packet.stream_name == 'previewout':
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])
            
            img_h = frame.shape[0]
            img_w = frame.shape[1]            
            cv2.imshow('previewout', frame)    
    if cv2.waitKey(1) == ord('q'):
        break
del p
depthai.deinit_device()
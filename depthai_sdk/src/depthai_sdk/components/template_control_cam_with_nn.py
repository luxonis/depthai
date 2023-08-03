${INIT}
width=xmax-xmin; height=ymax-ymin;
while True:
    detections = node.io['detections'].get()
    if len(detections.detections) == 0:
        continue
    largest_det = detections.detections[0]
    largest_size = 0
    for det in detections.detections:
        size = (det.xmax - det.xmin) * (det.ymax - det.ymin)
        if size > largest_size:
            largest_size = size
            largest_det = det
    det = largest_det
    ${DEBUG}node.warn(f"Detected ({det.xmin}, {det.ymin}) ({det.xmax}, {det.ymax})")
    ${RESIZE}
    if new_xmin < 0: new_xmin = 0.001
    if new_ymin < 0: new_ymin = 0.001
    if new_xmax > 1: new_xmax = 0.999
    if new_ymax > 1: new_ymax = 0.999

    ${DEBUG}node.warn(f"New ({new_xmin}, {new_ymin}) ({new_xmax}, {new_ymax})")
    ${DENORMALIZE}
    ${DEBUG}node.warn(f"Denormalized START ({startx}, {starty}) Width: {new_width}, height: {new_height})")
    control = CameraControl(1)
    ${CONTROL}
    node.io['control'].send(control)
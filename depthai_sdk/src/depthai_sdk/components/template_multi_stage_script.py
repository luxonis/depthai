import time
msgs = dict()

def add_msg(msg, name, seq = None):
    global msgs
    if seq is None:
        seq = msg.getSequenceNum()
    seq = str(seq)
    ${DEBUG}node.warn(f"New msg {name}, seq {seq}")

    # Each seq number has it's own dict of msgs
    if seq not in msgs:
        msgs[seq] = dict()
    msgs[seq][name] = msg

    # To avoid freezing (not necessary for this ObjDet model)
    if 15 < len(msgs):
        ${DEBUG}node.warn(f"Removing first element! len {len(msgs)}")
        msgs.popitem() # Remove first element

def get_msgs():
    global msgs
    seq_remove = [] # Arr of sequence numbers to get deleted
    for seq, syncMsgs in msgs.items():
        seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
        ${DEBUG}node.warn(f"Checking sync {seq}")

        # Check if we have both detections and color frame with this sequence number
        if len(syncMsgs) == 2: # 1 frame, 1 detection
            for rm in seq_remove:
                del msgs[rm]
            ${DEBUG}node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
            return syncMsgs # Returned synced msgs
    return None

def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.001
    if bb.ymin < 0: bb.ymin = 0.001
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999
    return bb

while True:
    time.sleep(0.001) # Avoid lazy looping

    frame = node.io['frames'].tryGet()
    if frame is not None:
        add_msg(frame, 'frames')

    dets = node.io['detections'].tryGet()
    if dets is not None:
        # TODO: in 2.18.0.0 use face_dets.getSequenceNum()
        passthrough = node.io['passthrough'].get()
        seq = passthrough.getSequenceNum()
        add_msg(dets, 'detections', seq)

    sync_msgs = get_msgs()
    if sync_msgs is not None:
        img = sync_msgs['frames']
        dets = sync_msgs['detections']
        for i, det in enumerate(dets.detections):
            ${CHECK_LABELS}
            cfg = ImageManipConfig()
            correct_bb(det)
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            ${DEBUG}node.warn(f"Sending {i + 1}. det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
            cfg.setResize(${WIDTH}, ${HEIGHT})
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)
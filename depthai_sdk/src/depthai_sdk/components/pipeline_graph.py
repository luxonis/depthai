#!/usr/bin/env python3
import re
import signal
import depthai as dai
from typing import Dict
from depthai_sdk.components.node_graph_qt import NodeGraph, BaseNode, PropertiesBinWidget
from depthai_sdk.components.node_graph_qt.constants import ViewerEnum
import time
from threading import Thread

class DepthaiNode(BaseNode):
    # unique node identifier.
    __identifier__ = 'dai'

    # initial default node name.
    NODE_NAME = 'Node'

    def __init__(self):
        super(DepthaiNode, self).__init__()

        # create QLineEdit text input widget.
        # self.add_text_input('my_input', 'Text Input', tab='widgets')


class PipelineGraph:

    def __init__(self, schema: Dict, device: dai.Device):

        from Qt import QtWidgets, QtCore

        node_color = {
            "ColorCamera": (241, 148, 138),
            "MonoCamera": (243, 243, 243),
            "ImageManip": (174, 214, 241),
            "VideoEncoder": (190, 190, 190),

            "NeuralNetwork": (171, 235, 198),
            "DetectionNetwork": (171, 235, 198),
            "MobileNetDetectionNetwork": (171, 235, 198),
            "MobileNetSpatialDetectionNetwork": (171, 235, 198),
            "YoloDetectionNetwork": (171, 235, 198),
            "YoloSpatialDetectionNetwork": (171, 235, 198),
            "SpatialDetectionNetwork": (171, 235, 198),

            "SPIIn": (242, 215, 213),
            "XLinkIn": (242, 215, 213),

            "SPIOut": (230, 176, 170),
            "XLinkOut": (230, 176, 170),

            "Script": (249, 231, 159),

            "StereoDepth": (215, 189, 226),
            "SpatialLocationCalculator": (215, 189, 226),

            "EdgeDetector": (248, 196, 113),
            "FeatureTracker": (248, 196, 113),
            "ObjectTracker": (248, 196, 113),
            "IMU": (248, 196, 113)
        }

        default_node_color = (190, 190, 190)  # For node types that does not appear in 'node_color'

        # handle SIGINT to make the app terminate on CTRL+C
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

        app = QtWidgets.QApplication(["DepthAI Pipeline Graph"])

        # create node graph controller.
        graph = NodeGraph()
        graph.set_background_color(255, 255, 255)
        graph.set_grid_mode(ViewerEnum.GRID_DISPLAY_NONE.value)

        graph.register_node(DepthaiNode)

        # create a node properties bin widget.
        properties_bin = PropertiesBinWidget(node_graph=graph)
        properties_bin.setWindowFlags(QtCore.Qt.Tool)

        # show the node properties bin widget when a node is double clicked.
        def display_properties_bin(node):
            global properties_bin
            if not properties_bin.isVisible():
                properties_bin.show()

        # wire function to "node_double_clicked" signal.
        graph.node_double_clicked.connect(display_properties_bin)

        # show the node graph widget.
        graph_widget = graph.widget
        graph_widget.resize(1100, 800)

        dai_connections = schema['connections']
        qt_nodes = {}
        dai_nodes = {}  # key = id, value = dict with keys 'type', 'blocking', 'queue_size' and 'name' (if args.use_variable_name)
        # Hold id->port
        input_port_map = dict()
        output_port_map = dict()
        input_name_to_id_map = dict()
        output_name_to_id_map = dict()

        for n in schema['nodes']:
            dict_n = n[1]
            node_name = dict_n['name']
            id = dict_n['id']
            dai_nodes[dict_n['id']] = {'type': node_name}
            dai_nodes[dict_n['id']]['name'] = f"{node_name} ({dict_n['id']})"
            # Create the node
            qt_nodes[id] = graph.create_node('dai.DepthaiNode', name=node_name, color=node_color.get(node_name, default_node_color), text_color=(0,0,0), push_undo=False)

            dict_n['ioInfo'] = list(sorted(dict_n['ioInfo'], key = lambda el: el[0][1]))
            for io in dict_n['ioInfo']:
                dict_io = io[1]
                io_id = dict_io['id']
                port_name = dict_io['name']
                port_group = dict_io['group']
                if port_group:
                    port_name = f"{dict_io['group']}[{port_name}]"
                blocking = dict_io['blocking']
                queue_size = dict_io['queueSize']
                port_color = (249,75,0) if blocking else (0,255,0)
                port_label = f"[{queue_size}] {port_name}"

                io_key = tuple([id, dict_io['group'], dict_io['name']])
                if dict_io['type'] == 3: # Input
                    input_port_map[dict_io['id']] = qt_nodes[id].add_input(name=port_label, color=port_color, multi_input=True)
                    input_name_to_id_map[io_key] = io_id
                elif dict_io['type'] == 0: # Output
                    output_port_map[dict_io['id']] = qt_nodes[id].add_output(name=port_name)
                    output_name_to_id_map[io_key] = io_id
                else:
                    print('Unhandled case!')

        # nodes = {} # First save all nodes and their inputs/outputs
        i = 0
        for c in dai_connections:
            src_node_id = c["node1Id"]
            src_name = c["node1Output"]
            src_group = c["node1OutputGroup"]
            dst_node_id = c["node2Id"]
            dst_name = c["node2Input"]
            dst_group = c["node2InputGroup"]

            out_key = tuple([src_node_id, src_group, src_name])
            in_key = tuple([dst_node_id, dst_group, dst_name])
            print(i,f"{out_key} -> {in_key}")

            output_port_map[output_name_to_id_map[out_key]].connect_to(input_port_map[input_name_to_id_map[in_key]], push_undo=False)
            i+=1

        # Lock the ports
        graph.lock_all_ports()

        graph_widget.show()
        graph.auto_layout_nodes()
        graph.fit_to_selection()
        graph.set_zoom(-0.9)
        graph.clear_selection()
        graph.clear_undo_stack()

        def traceEventReader(log_msg: dai.LogMessage):
            app.processEvents() # Process events
            # we are looking for  a line: EV:  ...
            match = re.search(r'EV:([0-9]+),S:([0-9]+),IDS:([0-9]+),IDD:([0-9]+),TSS:([0-9]+),TSN:([0-9]+)',
                              log_msg.payload.rstrip('\n'))
            if match:
                trace_event = TraceEvent()

                trace_event.event = int(match.group(1))
                trace_event.status = int(match.group(2))
                trace_event.src_id = int(match.group(3))
                trace_event.dst_id = int(match.group(4))
                trace_event.timestamp = int(match.group(5)) + (int(match.group(6)) / 1000000000.0)
                trace_event.host_timestamp = time.time()

                print('START->',log_msg.payload,'<-END')
                # buffer.append(trace_event)
                # buffer.sort(key=lambda event: event.timestamp)

        device.setLogLevel(dai.LogLevel.TRACE)
        device.addLogCallback(traceEventReader)
        class TraceEvent():
            event = 0
            status = 0
            src_id = 0
            dst_id = 0
            timestamp = 0.0
            host_timestamp = 0.0

        # while True:
        #     app.processEvents()
        #
        #     # TODO(themarpe) - move event processing to a separate function
        #     # Process trace events
        #
        #     # atleast 200ms should pass from latest event received
        #     if len(event_buffer) > 0 and time.time() - event_buffer[-1].host_timestamp > 0.2:
        #         # TODO(themarpe) - Process events
        #         pass
        #
        # app.exec_()

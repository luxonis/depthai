#!/usr/bin/env python3

import signal
from typing import Dict
from depthai_sdk.components.node_graph_qt import NodeGraph, BaseNode, PropertiesBinWidget
from depthai_sdk.components.node_graph_qt.constants import ViewerEnum


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

    def __init__(self, schema: Dict):

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
        dai_nodes = {}  # key = id, value = dict with keys 'type', 'blocking', 'queue_size' and 'name' (if args.use_variable_name)
        for n in schema['nodes']:
            dict_n = n[1]
            dai_nodes[dict_n['id']] = {'type': dict_n['name']}
            dai_nodes[dict_n['id']]['name'] = f"{dict_n['name']} ({dict_n['id']})"
            blocking = {}
            queue_size = {}
            for io in dict_n['ioInfo']:
                dict_io = io[1]
                port_name = dict_io['name']
                blocking[port_name] = dict_io['blocking']
                queue_size[port_name] = dict_io['queueSize']
            dai_nodes[dict_n['id']]['blocking'] = blocking
            dai_nodes[dict_n['id']]['queue_size'] = queue_size

        print("\nNodes (id):\n===========")
        for id in sorted(dai_nodes):
            print(f"{dai_nodes[id]['name']}")

        # create the nodes.
        qt_nodes = {}
        for id, node in dai_nodes.items():
            qt_nodes[id] = graph.create_node('dai.DepthaiNode', name=node['name'],
                                             color=node_color.get(node['type'], default_node_color),
                                             text_color=(0, 0, 0), push_undo=False)

        nodes = {} # First save all nodes and their inputs/outputs
        for c in dai_connections:
            src_node_id = c["node1Id"]
            src_node = qt_nodes[src_node_id]
            src_port_name = c["node1Output"]
            dst_node_id = c["node2Id"]
            dst_node = qt_nodes[dst_node_id]
            dst_port_name = c["node2Input"]

            if dst_node not in nodes:
                nodes[dst_node] = {'outputs': [], 'inputs': []}
            if src_node not in nodes:
                nodes[src_node] = {'outputs': [], 'inputs': []}

            dst_port_color = (249, 75, 0) if dai_nodes[dst_node_id]['blocking'][dst_port_name] else (0, 255, 0)
            dst_port_label = f"[{dai_nodes[dst_node_id]['queue_size'][dst_port_name]}] {dst_port_name}"

            nodes[dst_node]['inputs'].append({'name': dst_port_name, 'color': dst_port_color, 'label': dst_port_label})
            nodes[src_node]['outputs'].append({'name': src_port_name})

        for node, vals in nodes.items():
            # Go through all inputs/outputs, sort them by name (alphabetical order)
            vals['inputs'] = list(sorted(vals['inputs'], key = lambda el: el['name']))
            vals['outputs'] = list(sorted(vals['outputs'], key=lambda el: el['name']))
            # Create node input/output
            for output in vals['outputs']:
                if not output['name'] in list(node.outputs()):
                    node.add_output(name=output['name'])
            for input in vals['inputs']:
                if not input['name'] in list(node.inputs()):
                    node.add_input(name=input['label'], color=input['color'], multi_input=True)

        print("\nConnections:\n============")
        i = 0
        for c in dai_connections:
            src_node_id = c["node1Id"]
            src_node = qt_nodes[src_node_id]
            src_port_name = c["node1Output"]
            dst_node_id = c["node2Id"]
            dst_node = qt_nodes[dst_node_id]
            dst_port_name = c["node2Input"]
            dst_port_label = f"[{dai_nodes[dst_node_id]['queue_size'][dst_port_name]}] {dst_port_name}"
            # Create the connection between nodes
            print(i,
                  f"{dai_nodes[src_node_id]['name']}: {src_port_name} -> {dai_nodes[dst_node_id]['name']}: {dst_port_label}")
            src_node.outputs()[src_port_name].connect_to(dst_node.inputs()[dst_port_label], push_undo=False)
            i += 1

        # Lock the ports
        graph.lock_all_ports()

        graph_widget.show()
        graph.auto_layout_nodes()
        graph.fit_to_selection()
        graph.set_zoom(-0.9)
        graph.clear_selection()
        graph.clear_undo_stack()
        app.exec_()

CV2_HAS_GUI_SUPPORT = False

try:
    import cv2
    import re

    build_info = cv2.getBuildInformation()
    gui_support_regex = re.compile(r'GUI: +([A-Z]+)')
    gui_support_match = gui_support_regex.search(build_info)
    if gui_support_match:
        gui_support = gui_support_match.group(1)
        if gui_support.upper() != 'NONE':
            CV2_HAS_GUI_SUPPORT = True
except ImportError:
    pass

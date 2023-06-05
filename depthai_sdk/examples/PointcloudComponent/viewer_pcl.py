import depthai_viewer as viewer
viewer.init("Depthai Viewer")
viewer.connect()


viewer.log_rigid3(f"world", child_from_parent=([0, 0, 0], [1,0,0,0]), xyz="RDF")
viewer.log_pinhole("world/camera",
                  child_from_parent = [[f_len, 0,     u_cen],
                                       [0,     f_len, v_cen],
                                       [0,     0,     1  ]],
                  width = width,
                  height = height)

viewer.log_depth_image("world/camera/depth", [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
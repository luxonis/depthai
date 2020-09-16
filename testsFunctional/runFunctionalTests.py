import os
import consts.resource_paths

class TestManager:
    results = []

    def printTestName(self, testName):
        print("\033[1;34;40m----------------------------------------------------------------")
        print("Running " + testName)
        print("----------------------------------------------------------------\033[0m")


    def printResult(self, exitCode, testName):
        if exitCode == 0:
            print("\033[1;32;40m----------------------------------------------------------------")
            print(testName + " Passed!")
            print("----------------------------------------------------------------\033[0m")
        else:
            print("\033[1;31;40m----------------------------------------------------------------")
            print(testName + " Failed...")
            print("----------------------------------------------------------------\033[0m")

    def appendResult(self, result):
        self.results.append(result)

    def printSummary(self):
        for result in self.results:
            self.printResult(result[1], result[0])



if __name__ == "__main__":
    testMan = TestManager()
    testPath = consts.resource_paths.tests_functional_path

    # try data streams
    #meta_d2h
    result = os.system("python3 " + testPath + "test_meta_d2h.py -s " + "meta_d2h metaout")
    testMan.printResult(result, "meta_d2h")
    testMan.appendResult(["meta_d2h", result])

    #jpegout - if we want to do a functional test here, we need some way to trigger and detect a jpeg capture from the test.
    #result = os.system("python3 test_meta_d2h.py -s " + "previewout jpegout")
    #testMan.printResult(result, "jpegout")

    #object_tracker
    result = os.system("python3 " + testPath + "test_object_tracker.py -s " + "metaout object_tracker previewout")
    testMan.printResult(result, "object_tracker")
    print(result)
    testMan.appendResult(["object_tracker", result])


    # try visual streams.
    visualStreams = [
        "previewout",
        "left",
        "right",
        "depth",
        "disparity",
        "disparity_color"
    ]

    for stream in visualStreams:
        testMan.printTestName(stream)
        result = os.system("python3 " + testPath + "test_video.py -s " + stream)
        testMan.printResult(result, stream)
        testMan.appendResult([stream, result])


    #try various -cnn options
    cnndir = consts.resource_paths.nn_resource_path
    cnnlist = os.listdir(cnndir)
    for cnn in cnnlist:
        print(cnndir+cnn)
        if os.path.isdir(cnndir+cnn):
            testMan.printTestName(cnn)
            runStr = "python3 " + testPath + "test_video.py -cnn " + cnn
            result = os.system(runStr)
            testMan.printResult(result, cnn)
            testMan.appendResult([cnn, result])


    testMan.printSummary()


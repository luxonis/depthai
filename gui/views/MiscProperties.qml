import QtQuick 2.0
import QtQuick.Layouts 1.3
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

ListView {
    id: miscProperties
    delegate: Text {
        anchors.leftMargin: 50
        font.pointSize: 15
        horizontalAlignment: Text.AlignHCenter
        text: display
    }

    Rectangle {
        id: backgroundRect1
        color: "black"
        width: parent.width
        height: parent.height


        Text {
            id: text2
            x: 8
            y: 8
            width: 185
            height: 30
            color: "#ffffff"
            text: qsTr("Recording")
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            font.family: "Courier"
            font.styleName: "Regular"
        }

        TextField {
            id: encColorFps
            x: 140
            y: 44
            width: 83
            height: 27
            bottomPadding: 7
            validator: IntValidator {}
            placeholderText: qsTr("FPS")
            onEditingFinished: {
                appBridge.toggleColorEncoding(encColorSwitch.checked, encColorFps.text)
            }
        }

        Switch {
            id: encColorSwitch
            x: 8
            y: 44
            width: 150
            height: 27
            text: qsTr("<font color=\"white\">Color</font>")
            bottomPadding: 5
            onToggled: {
                appBridge.toggleColorEncoding(encColorSwitch.checked, encColorFps.text)
            }
        }

        TextField {
            enabled: depthEnabled
            id: encLeftFps
            x: 140
            y: 77
            width: 83
            height: 27
            bottomPadding: 7
            validator: IntValidator {}
            placeholderText: qsTr("FPS")
            onEditingFinished: {
                appBridge.toggleLeftEncoding(encLeftSwitch.checked, encLeftFps.text)
            }
        }

        Switch {
            enabled: depthEnabled
            id: encLeftSwitch
            x: 8
            y: 77
            width: 150
            height: 27
            text: qsTr("<font color=\"white\">Left</font>")
            bottomPadding: 5
            onToggled: {
                appBridge.toggleLeftEncoding(encLeftSwitch.checked, encLeftFps.text)
            }
        }

        TextField {
            enabled: depthEnabled
            id: encRightFps
            x: 140
            y: 110
            width: 83
            height: 27
            bottomPadding: 7
            validator: IntValidator {}
            placeholderText: qsTr("FPS")
            onEditingFinished: {
                appBridge.toggleRightEncoding(encRightSwitch.checked, encRightFps.text)
            }
        }

        Switch {
            enabled: depthEnabled
            id: encRightSwitch
            x: 8
            y: 110
            width: 150
            height: 27
            text: qsTr("<font color=\"white\">Right</font>")
            bottomPadding: 5
            onToggled: {
                appBridge.toggleRightEncoding(encLeftSwitch.checked, encLeftFps.text)
            }
        }

        Text {
            id: text3
            x: 8
            y: 203
            width: 185
            height: 30
            color: "#ffffff"
            text: qsTr("Reporting")
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            font.family: "Courier"
            font.styleName: "Regular"
        }

        Switch {
            id: tempSwitch
            x: 8
            y: 239
            width: 185
            height: 27
            text: qsTr("<font color=\"white\">Temperature</font>")
            bottomPadding: 5
            onToggled: {
                appBridge.selectReportingOptions(tempSwitch.checked, cpuSwitch.checked, memSwitch.checked)
            }
        }

        Switch {
            id: cpuSwitch
            x: 8
            y: 272
            width: 185
            height: 27
            text: qsTr("<font color=\"white\">CPU</font>")
            bottomPadding: 5
            onToggled: {
                appBridge.selectReportingOptions(tempSwitch.checked, cpuSwitch.checked, memSwitch.checked)
            }
        }

        Switch {
            id: memSwitch
            x: 8
            y: 305
            width: 185
            height: 27
            text: qsTr("<font color=\"white\">Memory</font>")
            bottomPadding: 5
            onToggled: {
                appBridge.selectReportingOptions(tempSwitch.checked, cpuSwitch.checked, memSwitch.checked)
            }
        }

        TextField {
            id: textField3
            x: 110
            y: 352
            width: 170
            height: 27
            bottomPadding: 7
            placeholderText: qsTr("/path/to/report.csv")
            onEditingFinished: {
                appBridge.selectReportingPath(text)
            }
        }

        Text {
            id: text26
            x: 8
            y: 352
            width: 90
            height: 27
            color: "#ffffff"
            text: qsTr("Destination")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        TextField {
            id: textField4
            x: 116
            y: 150
            width: 170
            height: 27
            bottomPadding: 7
            placeholderText: qsTr("/path/to/output/directory/")
            onEditingFinished: {
                appBridge.selectEncodingPath(text)
            }
        }

        Text {
            id: text27
            x: 14
            y: 150
            width: 90
            height: 27
            color: "#ffffff"
            text: qsTr("Destination")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Text {
            id: textOptions1
            x: 350
            y: 203
            width: 185
            height: 30
            color: "#ffffff"
            text: qsTr("<font color=\"white\">Applications</font>")
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.family: "Courier"
        }

        Text {
            id: uvcLabel
            x: 330
            y: 250
            width: 141
            height: 27
            color: "#ffffff"
            text: qsTr("<font color=\"white\">UVC Mode (Webcam)</font>")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Button {
            id: uvcButton
            x: 477
            y: 250
            width: 100
            height: 27
            text: runningApp === "uvc" ? qsTr("Terminate") : qsTr("Run")
            onClicked: runningApp === "uvc" ? appBridge.terminateApp("uvc") : appBridge.runApp("uvc")
        }
    }
}
/*##^##
Designer {
    D{i:0;autoSize:true;formeditorZoom:2;height:480;width:640}
}
##^##*/

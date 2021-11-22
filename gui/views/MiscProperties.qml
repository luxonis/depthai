import QtQuick 2.0
import QtQuick.Layouts 1.11
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
            x: 110
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
            width: 96
            height: 27
            text: qsTr("Color")
            bottomPadding: 5
            onToggled: {
                appBridge.toggleColorEncoding(encColorSwitch.checked, encColorFps.text)
            }
        }

        TextField {
            enabled: depthEnabled
            id: encLeftFps
            x: 110
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
            width: 96
            height: 27
            text: qsTr("Left")
            bottomPadding: 5
            onToggled: {
                appBridge.toggleLeftEncoding(encLeftSwitch.checked, encLeftFps.text)
            }
        }

        TextField {
            enabled: depthEnabled
            id: encRightFps
            x: 110
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
            width: 96
            height: 27
            text: qsTr("Right")
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
            text: qsTr("Temperature")
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
            text: qsTr("CPU")
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
            text: qsTr("Memory")
            bottomPadding: 5
            onToggled: {
                appBridge.selectReportingOptions(tempSwitch.checked, cpuSwitch.checked, memSwitch.checked)
            }
        }

        TextField {
            id: textField3
            x: 110
            y: 352
            width: 227
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
            width: 227
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
    }
}
/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}D{i:11}D{i:12}D{i:13}D{i:14}
}
##^##*/

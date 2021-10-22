import QtQuick 2.0
import QtQuick.Layouts 1.11
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

ListView {
    id: listView

    Rectangle {
        id: backgroundRect1
        color: "black"
        width: parent.width
        height: parent.height

        Text {
            id: text10
            x: 79
            y: 8
            width: 433
            height: 30
            color: "#ffffff"
            text: qsTr("Camera Properties")
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.family: "Courier"
        }

        Text {
            id: text11
            x: 0
            y: 44
            width: 197
            height: 30
            color: "#ffffff"
            text: qsTr("Color")
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.family: "Courier"
        }

        Text {
            id: text12
            x: 385
            y: 44
            width: 197
            height: 30
            color: "#ffffff"
            text: qsTr("Right")
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.family: "Courier"
        }

        Text {
            id: text13
            x: 203
            y: 44
            width: 181
            height: 30
            color: "#ffffff"
            text: qsTr("Left")
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.family: "Courier"
        }

        TextField {
            id: textField
            x: 85
            y: 80
            width: 106
            height: 25
            text: ""
            bottomPadding: 5
            validator: IntValidator {}
            placeholderText: "ISO"
            font.family: "Courier"
            onEditingFinished: {
                colorCamBridge.setIsoExposure(text, textField1.text)
            }
        }

        Text {
            id: text14
            x: -6
            y: 80
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("ISO")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }
        
        TextField {
            id: textField1
            x: 85
            y: 111
            width: 106
            height: 25
            text: ""
            bottomPadding: 5
            font.family: "Courier"
            placeholderText: qsTr("Exposure")
            validator: IntValidator {}
            onEditingFinished: {
                colorCamBridge.setIsoExposure(textField.text, text)
            }
        }


        Text {
            id: text15
            x: -6
            y: 111
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Exposure")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider2
            x: 85
            y: 142
            width: 106
            height: 25
            stepSize: 1
            to: 10
            from: -10
            value: 0
            onValueChanged: {
                colorCamBridge.setSaturation(value)
            }
        }

        Text {
            id: text16
            x: -6
            y: 142
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Saturation")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider3
            x: 85
            y: 173
            width: 106
            height: 25
            stepSize: 1
            to: 10
            from: -10
            value: 0
            onValueChanged: {
                colorCamBridge.setContrast(value)
            }
        }

        Text {
            id: text17
            x: -6
            y: 173
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Contrast")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider4
            x: 85
            y: 204
            width: 106
            height: 25
            stepSize: 1
            to: 10
            from: -10
            value: 0
            onValueChanged: {
                colorCamBridge.setBrightness(value)
            }
        }

        Text {
            id: text18
            x: -6
            y: 204
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Brightness")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider5
            x: 85
            y: 235
            width: 106
            height: 25
            stepSize: 1
            to: 4
            from: 0
            value: 0
            onValueChanged: {
                colorCamBridge.setSharpness(value)
            }
        }

        Text {
            id: text19
            x: -6
            y: 235
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Sharpness")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Text {
            id: text20
            x: 393
            y: 80
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("ISO")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        TextField {
            id: textField2
            x: 484
            y: 80
            width: 106
            height: 25
            color: "#ddffffff"
            text: ""
            bottomPadding: 5
            validator: IntValidator {}
            placeholderText: "ISO"
            font.family: "Courier"
            onEditingFinished: {
                rightCamBridge.setIsoExposure(text, textField3.text)
            }
        }

        TextField {
            id: textField3
            x: 484
            y: 111
            width: 106
            height: 25
            color: "#ddffffff"
            text: ""
            bottomPadding: 5
            font.family: "Courier"
            placeholderText: qsTr("Exposure")
            validator: IntValidator {}
            onEditingFinished: {
                rightCamBridge.setIsoExposure(textField2.text, text)
            }
        }

        Text {
            id: text21
            x: 393
            y: 111
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Exposure")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider6
            x: 484
            y: 142
            width: 106
            height: 25
            stepSize: 1
            to: 10
            from: -10
            value: 0
            onValueChanged: {
                rightCamBridge.setSaturation(value)
            }
        }

        Text {
            id: text22
            x: 393
            y: 142
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Saturation")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider7
            x: 484
            y: 173
            width: 106
            height: 25
            stepSize: 1
            to: 10
            from: -10
            value: 0
            onValueChanged: {
                rightCamBridge.setContrast(value)
            }
        }

        Text {
            id: text23
            x: 393
            y: 173
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Contrast")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider8
            x: 484
            y: 204
            width: 106
            height: 25
            stepSize: 1
            to: 10
            from: -10
            value: 0
            onValueChanged: {
                rightCamBridge.setBrightness(value)
            }
        }

        Text {
            id: text24
            x: 393
            y: 204
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Brightness")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider9
            x: 484
            y: 235
            width: 106
            height: 25
            stepSize: 1
            to: 4
            from: 0
            value: 0
            onValueChanged: {
                rightCamBridge.setSharpness(value)
            }
        }

        Text {
            id: text25
            x: 393
            y: 235
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Sharpness")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Text {
            id: text26
            x: 197
            y: 80
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("ISO")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        TextField {
            id: textField4
            x: 288
            y: 80
            width: 106
            height: 25
            color: "#ddffffff"
            text: ""
            bottomPadding: 5
            placeholderText: "ISO"
            font.family: "Courier"
            validator: IntValidator {}
            onEditingFinished: {
                leftCamBridge.setIsoExposure(text, textField5.text)
            }
        }

        TextField {
            id: textField5
            x: 288
            y: 111
            width: 106
            height: 25
            color: "#ddffffff"
            text: ""
            bottomPadding: 5
            placeholderText: qsTr("Exposure")
            validator: IntValidator {}
            onEditingFinished: {
                leftCamBridge.setIsoExposure(textField4.text, text)
            }
        }

        Text {
            id: text27
            x: 197
            y: 111
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Exposure")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider10
            x: 288
            y: 142
            width: 106
            height: 25
            stepSize: 1
            to: 10
            from: -10
            value: 0
            onValueChanged: {
                leftCamBridge.setSaturation(value)
            }
        }

        Text {
            id: text28
            x: 197
            y: 142
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Saturation")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider11
            x: 288
            y: 173
            width: 106
            height: 25
            stepSize: 1
            to: 10
            from: -10
            value: 0
            onValueChanged: {
                leftCamBridge.setContrast(value)
            }
        }

        Text {
            id: text29
            x: 197
            y: 173
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Contrast")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider12
            x: 288
            y: 204
            width: 106
            height: 25
            stepSize: 1
            to: 10
            from: -10
            value: 0
            onValueChanged: {
                leftCamBridge.setBrightness(value)
            }
        }

        Text {
            id: text30
            x: 197
            y: 204
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Brightness")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }

        Slider {
            id: slider13
            x: 288
            y: 235
            width: 106
            height: 25
            stepSize: 1
            to: 4
            from: 0
            value: 0
            onValueChanged: {
                leftCamBridge.setSharpness(value)
            }
        }

        Text {
            id: text31
            x: 197
            y: 235
            width: 90
            height: 25
            color: "#ffffff"
            text: qsTr("Sharpness")
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Courier"
        }
    }
}
/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
##^##*/

import QtQuick 2.0
import QtQuick.Layouts 1.11
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

ListView {
    id: view
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

    ComboBox {
        id: comboBox
        x: 0
        y: 102
        width: 195
        height: 33
        model: medianChoices
        onActivated: function(index) {
            depthBridge.setMedianFilter(model[index])
        }
    }

    Slider {
        id: slider
        x: 359
        y: 102
        width: 200
        height: 25
        snapMode: RangeSlider.NoSnap
        stepSize: 1
        from: 0
        to: 255
        value: 240
        onValueChanged: {
            depthBridge.setDisparityConfidenceThreshold(value)
        }
    }

    Text {
        id: text2
        x: 0
        y: 71
        width: 195
        height: 25
        color: "#ffffff"
        text: qsTr("Median filtering")
        font.pixelSize: 18
        font.styleName: "Regular"
        font.weight: Font.Medium
        font.family: "Courier"
    }

    Switch {
        id: switch1
        x: 0
        y: 187
        text: qsTr("Left Right Check")
        transformOrigin: Item.Center
        font.preferShaping: false
        font.kerning: false
        font.family: "Courier"
        autoExclusive: false
        onToggled: {
            depthBridge.toggleLeftRightCheck(switch1.checked)
        }
    }

    Switch {
        id: switch2
        x: 0
        y: 233
        text: qsTr("Extended Disparity")
        autoExclusive: false
        font.kerning: false
        font.family: "Courier"
        font.preferShaping: false
        transformOrigin: Item.Center
        onToggled: {
            depthBridge.toggleExtendedDisparity(switch2.checked)
        }
    }

    Switch {
        id: switch3
        x: 0
        y: 141
        text: qsTr("Subpixel")
        autoExclusive: false
        font.kerning: false
        transformOrigin: Item.Center
        font.preferShaping: false
        font.family: "Courier"
        onToggled: {
            depthBridge.toggleSubpixel(switch3.checked)
        }
    }

    Text {
        id: text3
        x: 359
        y: 71
        width: 200
        height: 25
        color: "#ffffff"
        text: qsTr("Confidence Threshold")
        font.pixelSize: 18
        horizontalAlignment: Text.AlignHCenter
        font.styleName: "Regular"
        font.weight: Font.Medium
        font.family: "Courier"
    }

    Slider {
        id: slider1
        x: 360
        y: 172
        width: 200
        height: 25
        stepSize: 1
        snapMode: RangeSlider.NoSnap
        value: 240
        to: 255
        onValueChanged: {
            depthBridge.setBilateralSigma(value)
        }
    }

    Text {
        id: text4
        x: 339
        y: 102
        width: 14
        height: 25
        color: "#ffffff"
        text: qsTr("0")
        font.pixelSize: 12
    }

    Text {
        id: text5
        x: 566
        y: 102
        width: 17
        height: 25
        color: "#ffffff"
        text: qsTr("255")
        font.pixelSize: 12
    }

    Text {
        id: text6
        x: 359
        y: 141
        width: 200
        height: 25
        color: "#ffffff"
        text: qsTr("Bilateral Sigma")
        font.pixelSize: 18
        horizontalAlignment: Text.AlignHCenter
        font.styleName: "Regular"
        font.weight: Font.Medium
        font.family: "Courier"
    }

    Text {
        id: text7
        x: 338
        y: 172
        width: 17
        height: 25
        color: "#ffffff"
        text: qsTr("0")
        font.pixelSize: 12
    }

    Text {
        id: text8
        x: 566
        y: 175
        width: 17
        height: 20
        color: "#ffffff"
        text: qsTr("255")
        font.pixelSize: 12
        rotation: 0
    }

    Text {
        id: text9
        x: 359
        y: 209
        width: 200
        height: 25
        color: "#ffffff"
        text: qsTr("Depth Range")
        font.pixelSize: 18
        horizontalAlignment: Text.AlignHCenter
        font.styleName: "Regular"
        font.weight: Font.Medium
        font.family: "Courier"
    }

    RangeSlider {
        id: rangeSlider
        x: 362
        y: 233
        width: 198
        height: 27
        snapMode: RangeSlider.NoSnap
        stepSize: 1
        to: 10000
        focusPolicy: Qt.StrongFocus
        second.value: 10000
        first.value: 0
        first.onMoved: {
            depthBridge.setDepthRange(first.value, second.value)
        }
        second.onMoved: {
            depthBridge.setDepthRange(first.value, second.value)
        }
    }

    Text {
        id: text1
        x: 79
        y: 0
        width: 433
        height: 30
        color: "#ffffff"
        text: qsTr("Depth Properties")
        font.pixelSize: 26
        horizontalAlignment: Text.AlignHCenter
        font.styleName: "Regular"
        font.family: "Courier"
    }

    Text {
        id: text32
        x: 566
        y: 237
        width: 17
        height: 20
        color: "#ffffff"
        text: qsTr("10m")
        font.pixelSize: 12
        rotation: 0
    }

    Text {
        id: text33
        x: 338
        y: 237
        width: 17
        height: 20
        color: "#ffffff"
        text: qsTr("0m")
        font.pixelSize: 12
        rotation: 0
    }

    }
}
/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
##^##*/

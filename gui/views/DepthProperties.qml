import QtQuick 2.0
import QtQuick.Layouts 1.11
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

ListView {
    id: depthProperties
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
            id: dctSlider
            x: 360
            y: 89
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
            id: switch5
            x: 328
            y: 0
            width: 167
            height: 38
            text: qsTr("Enabled")
            autoExclusive: false
            font.family: "Courier"
            checked: true
            font.kerning: false
            transformOrigin: Item.Center
            font.preferShaping: false
            onToggled: {
                appBridge.toggleDepth(switch5.checked)
            }
        }

        Switch {
            enabled: false
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
            id: sigmaSlider
            x: 362
            y: 133
            width: 200
            height: 25
            stepSize: 1
            snapMode: RangeSlider.NoSnap
            value: 0
            to: 255
            onValueChanged: {
                depthBridge.setBilateralSigma(value)
            }
        }

        Text {
            id: text5
            x: 566
            y: 95
            width: 17
            height: 25
            color: "#ffffff"
            text: dctSlider.value
            font.pixelSize: 12
        }

        Text {
            id: text6
            x: 360
            y: 115
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
            id: text8
            x: 566
            y: 136
            width: 17
            height: 20
            color: "#ffffff"
            text: sigmaSlider.value
            font.pixelSize: 12
            rotation: 0
        }

        Text {
            id: text9
            x: 362
            y: 158
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
            id: depthRangeSlider
            x: 364
            y: 181
            width: 198
            height: 27
            snapMode: RangeSlider.NoSnap
            stepSize: 100
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
            x: 44
            y: 4
            width: 285
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
            y: 185
            width: 17
            height: 20
            color: "#ffffff"
            text: (depthRangeSlider.second.value / 1000).toFixed(1) + "m"
            font.pixelSize: 12
            rotation: 0
        }

        Text {
            id: text33
            x: 337
            y: 185
            width: 17
            height: 20
            color: "#ffffff"
            text: (depthRangeSlider.first.value / 1000).toFixed(1) + "m"
            font.pixelSize: 12
            rotation: 0
        }

        Text {
            id: text10
            x: 360
            y: 214
            width: 200
            height: 25
            color: "#ffffff"
            text: qsTr("LRC Threshold")
            font.pixelSize: 18
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.weight: Font.Medium
            font.family: "Courier"
        }

        Slider {
            id: lrcSlider
            x: 361
            y: 233
            width: 198
            height: 27
            stepSize: 1
            to: 10
            value: 10
            from: 0
            onValueChanged: {
                depthBridge.setLrcThreshold(value)
            }
        }

        Text {
            id: text34
            x: 566
            y: 233
            width: 17
            height: 20
            color: "#ffffff"
            text: lrcSlider.value
            font.pixelSize: 12
            rotation: 0
        }

        Text {
            id: text35
            x: 337
            y: 233
            width: 17
            height: 20
            color: "#ffffff"
            font.pixelSize: 12
            rotation: 0
        }

        Switch {
            id: switch6
            x: 443
            y: 0
            width: 169
            height: 38
            text: qsTr("Use Disparity")
            autoExclusive: false
            font.family: "Courier"
            font.kerning: false
            transformOrigin: Item.Center
            font.preferShaping: false
            onToggled: {
                appBridge.toggleDisparity(switch6.checked)
            }
        }


    }
}
/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}D{i:7}D{i:24}
}
##^##*/

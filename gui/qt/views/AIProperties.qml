import QtQuick 2.0
import QtQuick.Layouts 1.3
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

ListView {
    id: aiProperties

    Rectangle {
        id: backgroundRectAI
        color: "black"
        z: 0
        width: parent.width
        height: parent.height

        Text {
            id: textAI
            x: 79
            y: 8
            width: 322
            height: 30
            color: "#ffffff"
            text: qsTr("AI Properties")
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            font.styleName: "Regular"
            font.family: "Courier"
        }

        Rectangle {
            id: aiRect

            Text {
                id: text33
                x: 90
                y: 56
                width: 124
                height: 33
                color: "#ffffff"
                text: qsTr("CNN model")
                font.pixelSize: 12
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                font.weight: Font.Bold
                font.family: "Courier"
            }

            ComboBox {
                id: comboBoxModels
                x: 220
                y: 56
                width: 282
                height: 33
                model: modelChoices
                onActivated: function(index) {
                    aiBridge.setCnnModel(model[index])
                }
            }

            Text {
                id: text34
                x: 90
                y: 134
                width: 124
                height: 33
                color: "#ffffff"
                text: qsTr("Model source")
                font.pixelSize: 12
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                font.family: "Courier"
                font.weight: Font.Bold
            }

            ComboBox {
                id: comboBox1
                x: 220
                y: 134
                width: 141
                height: 33
                model: modelSourceChoices
                onActivated: function(index) {
                    aiBridge.setModelSource(model[index])
                }
            }

            Text {
                id: text36
                x: 90
                y: 95
                width: 124
                height: 33
                color: "#ffffff"
                text: qsTr("SHAVEs")
                font.pixelSize: 12
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                font.family: "Courier"
                font.weight: Font.Bold
            }

            Slider {
                id: lrcSlider
                x: 221
                y: 95
                width: 261
                height: 33
                value: 6
                stepSize: 1
                onValueChanged: {
                    aiBridge.setShaves(value)
                }
                to: 16
                from: 1
            }

            Text {
                id: text37
                x: 482
                y: 101
                width: 18
                height: 20
                color: "#ffffff"
                text: lrcSlider.value
                font.pixelSize: 12
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                rotation: 0
            }

            Text {
                id: text43
                x: 403
                y: 134
                width: 124
                height: 33
                color: "#ffffff"
                text: qsTr("Full FOV")
                font.pixelSize: 12
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignVCenter
                font.family: "Courier"
                font.weight: Font.Bold
            }

            CheckBox {
                id: checkBoxFullFov
                x: 369
                y: 137
                width: 37
                height: 23
                checked: true
                onClicked: aiBridge.setFullFov(checkBoxFullFov.checked)
            }
        }

        CheckBox {
            id: aiAdvancedSwitch
            x: 167
            y: 228
            text: qsTr("<font color=\"white\">Show advanced options</font>")
            font.family: "Courier"
            autoExclusive: false
            transformOrigin: Item.Center
            font.pointSize: 21
            checked: false
        }

        Rectangle {
            id: aiAdvancedRect

            states: State {
                name: "hidden"; when: !aiAdvancedSwitch.checked
                PropertyChanges { target: aiAdvancedRect; opacity: 0 }
            }

            Text {
                id: text41
                x: 90
                y: 288
                width: 124
                height: 33
                color: "#ffffff"
                text: qsTr("OpenVINO version")
                font.pixelSize: 12
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                font.family: "Courier"
                font.weight: Font.Bold
            }

            ComboBox {
                id: comboBox2
                x: 220
                y: 288
                width: 170
                height: 33
                model: ovVersions
                onActivated: function(index) {
                    aiBridge.setOvVersion(model[index])
                }
            }

            Text {
                id: text42
                x: 90
                y: 327
                width: 124
                height: 33
                color: "#ffffff"
                text: qsTr("Label to count")
                font.pixelSize: 12
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                font.family: "Courier"
                font.weight: Font.Bold
            }

            ComboBox {
                id: comboBox3
                x: 220
                y: 327
                width: 141
                height: 33
                model: countLabels
                onActivated: function(index) {
                    aiBridge.setCountLabel(model[index])
                }
            }

            Switch {
                enabled: depthEnabled
                id: switch3
                x: 90
                y: 375
                width: 250
                height: 48
                text: qsTr("<font color=\"white\">Spatial Bounding Boxes</font>")
                font.family: "Courier"
                autoExclusive: false
                transformOrigin: Item.Center
                onToggled: {
                    aiBridge.setSbb(switch3.checked)
                }
            }

            Text {
                id: text40
                x: 320
                y: 383
                width: 106
                height: 33
                color: "#ffffff"
                text: qsTr("SBB factor")
                font.pixelSize: 12
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                font.family: "Courier"
                font.weight: Font.Bold
            }

            Slider {
                enabled: depthEnabled
                id: sbbFactorSlider
                x: 423
                y: 383
                width: 96
                height: 33
                value: 0.3
                stepSize: 0.1
                to: 1
                from: 0.1
                onValueChanged: {
                    aiBridge.setSbbFactor(value)
                }
            }

            Text {
                id: text39
                x: 527
                y: 389
                width: 18
                height: 20
                color: "#ffffff"
                text: sbbFactorSlider.value.toFixed(1)
                font.pixelSize: 12
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                rotation: 0
            }
        }

        Switch {
            id: switch5
            x: 359
            y: 0
            width: 167
            height: 38
            text: qsTr("<font color=\"white\">Enabled</font>")
            checked: true
            autoExclusive: false
            font.family: "Courier"
            transformOrigin: Item.Center
            onToggled: {
                appBridge.toggleNN(switch5.checked)
            }
        }

    }
}
/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}D{i:26}
}
##^##*/

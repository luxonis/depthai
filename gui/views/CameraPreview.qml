import QtQuick 2.0
import QtQuick.Layouts 1.3
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

import dai.gui 1.0

ListView {
    id: cameraPreview

    Rectangle {
        id: cameraPreviewRect
        color: "black"
        width: parent.width
        height: 640

        ComboBox {
            id: comboBoxImage
            x: 100
            y: 5
            width: 150
            height: 30
            model: previewChoices
            onActivated: function(index) {
                previewBridge.changeSelected(model[index])
            }
        }

        ComboBox {
            id: comboBoxDevices
            x: 260
            y: 5
            width: 200
            height: 30
            model: deviceChoices
            onActivated: function(index) {
                appBridge.selectDevice(model[index])
            }
        }

        Button {
            x: 470
            y: 5
            height: 30
            width: 100
            text: "Reload"
            onClicked: appBridge.reloadDevices()
        }

        ImageWriter {
            id: imageWriter
            objectName: "writer"
            x: 40
            y: 40
            width: parent.width - 80
            height: parent.height - 80
        }
    }
}
/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
##^##*/

import QtQuick 2.0
import QtQuick.Layouts 1.11
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
            x: 210
            y: 5
            width: 200
            height: 30
            model: previewChoices
            onActivated: function(index) {
                previewBridge.changeSelected(model[index])
            }
        }

        ImageWriter {
            id: imageWriter
            x: 40
            y: 40
            width: 560
            height: 560
        }
    }
}
/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
##^##*/

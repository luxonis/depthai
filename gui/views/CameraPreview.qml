import QtQuick 2.0
import QtQuick.Layouts 1.11
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

ListView {
    id: cameraPreview

    Rectangle {
        id: cameraPreviewRect
        color: "black"
        width: parent.width
        height: parent.height
    }
}
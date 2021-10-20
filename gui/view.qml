/****************************************************************************
**
** Copyright (C) 2021 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of Qt for Python.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

import QtQuick 2.0
import QtQuick.Layouts 1.11
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

ApplicationWindow {
    width: 640
    height: 640
    Material.theme: Material.Dark
    Material.accent: Material.Red
    visible: true

    Bridge {
        id: bridge
    }
    
    Rectangle {
        id: root
        x: 0
        y: 0
        width: parent.width
        height: 640
        color: "#000000"
        enabled: true
        
        ListView {
            id: view
            anchors.fill: root
            anchors.margins: 25
            anchors.bottomMargin: 320
            delegate: Text {
                anchors.leftMargin: 50
                font.pointSize: 15
                horizontalAlignment: Text.AlignHCenter
                text: display
            }
            
            ComboBox {
                id: comboBox
                x: 0
                y: 110
                width: 195
                height: 25
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
                    bridge.setDisparityConfidenceThreshold(value)
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
                autoExclusive: true
            }
            
            Switch {
                id: switch2
                x: 0
                y: 233
                text: qsTr("Extended Disparity")
                autoExclusive: true
                font.kerning: false
                font.family: "Courier"
                font.preferShaping: false
                transformOrigin: Item.Center
                onToggled: {
                    bridge.toggleSubpixel(switch2.checked)
                }
            }
            
            Switch {
                id: switch3
                x: 0
                y: 141
                text: qsTr("Subpixel")
                autoExclusive: true
                font.kerning: false
                transformOrigin: Item.Center
                font.preferShaping: false
                font.family: "Courier"
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
        
        ListView {
            id: listView
            x: 25
            y: 348
            width: 590
            height: 284

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

            TextField {
                id: textField
                x: 85
                y: 80
                width: 106
                height: 25
                text: ""
                placeholderText: "ISO"
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
                placeholderText: qsTr("Exposure")
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
                text: ""
                placeholderText: "ISO"
                font.family: "Courier"
            }

            TextField {
                id: textField3
                x: 484
                y: 111
                width: 106
                height: 25
                text: ""
                font.family: "Courier"
                placeholderText: qsTr("Exposure")
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
                text: ""
                placeholderText: "ISO"
                font.family: "Courier"
            }

            TextField {
                id: textField5
                x: 288
                y: 111
                width: 106
                height: 25
                text: ""
                placeholderText: qsTr("Exposure")
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
}

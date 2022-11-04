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
import QtQuick.Layouts 1.3
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

import dai.gui 1.0

ApplicationWindow {
    width: 1270
    height: 640
    Material.theme: Material.Dark
    Material.accent: Material.Red
    visible: true

    property var previewChoices
    property var modelChoices
    property var modelSourceChoices
    property var ovVersions
    property var countLabels
    property var medianChoices
    property var colorResolutionChoices
    property var monoResolutionChoices
    property var restartRequired
    property var deviceChoices
    property var irEnabled: false
    property var irDotBrightness: 0
    property var irFloodBrightness: 0
    property var depthEnabled: true
    property var statisticsAccepted: true
    property var runningApp

    property bool lrc: false

    AppBridge {
        id: appBridge
    }

    DepthBridge {
        id: depthBridge
    }
    ColorCamBridge {
        id: colorCamBridge
    }
    MonoCamBridge {
        id: monoCamBridge
    }
    PreviewBridge {
        id: previewBridge
    }
    AIBridge {
        id: aiBridge
    }

    Rectangle {
        id: root
        x: 0
        y: 0
        width: parent.width
        height: parent.height
        color: "#000000"
        enabled: true

        CameraPreview {
          x: 0
          y: 0
          width: parent.width - 630
          height: parent.height
        }

        TabBar {
            id: bar
            x: parent.width - 630
            y: 0
            height: 50
            width: 590

            TabButton {
                text: "AI"
            }

            TabButton {
                enabled: depthEnabled
                text: "Depth"
            }
            TabButton {
               text: "Camera"
            }
            TabButton {
               text: "Misc"
            }
        }

        StackLayout {
          x: parent.width - 630
          y: 70
          width: 630
          currentIndex: bar.currentIndex
          Item {
                AIProperties {}
          }
          Item {
                DepthProperties {}
          }
          Item {
               CameraProperties {}
          }
          Item {
               MiscProperties {}
          }
        }

        Button {
            x: parent.width - 600
            y: 540
            enabled: restartRequired || false
            height: 60
            width: 563
            text: "Apply and Restart"
            onClicked: appBridge.applyAndRestart()
        }
    }
}

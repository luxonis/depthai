# DepthAI Demo GUI


## Local development

> :warning: Instructions tested on Mac but should be similar in other OSes. Feel free to extend this section if it's different in your scenario :warning:

First, download QT Everywhere 6.2.0 from [this link](https://download.qt.io/archive/qt/6.2/6.2.0/single/qt-everywhere-src-6.2.0.tar.xz.mirrorlist) (on Windows, use [this link](https://download.qt.io/archive/qt/6.2/6.2.0/single/qt-everywhere-src-6.2.0.zip.mirrorlist))

Next, extract the package and cd into it. Now, run the following commands:

```
# to prepare qt everywhere repository
$ ./configure -prefix $PWD/qtbase
# to compile the qt, may take a while (on my MacBook Air it took 4hrs)
$ cmake --build .
```

Now, download the [QT Creator](https://www.qt.io/product/development-tools). After downloading and installing this tool:
- go to Preferences > Kits > Qt Versions
- Click "Add"
- point to the qt everywhere directory and `qtbase/bin/qmake`
- Go to Kits
- Edit both of the kits by clicking on them, scrolling down to Qt Version and selecting the qt-everywhere version we just added
- Restart QT Creator
- go to Preferences > Qt Quick
- Click "Qt Quick Designer"
- Under "QML Emulation Layer" select second option - "Use QML emulation layer that is built...", leaving the default path
- Close Preferences
- Open .qml file and click "Design"
- A build process should start automatically. It may throw a warning that the build process is not responding, ignore
- Restart QT Creator

Now, the setup is ready and the Designer tool can be used too

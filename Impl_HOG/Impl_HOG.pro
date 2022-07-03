QT += core
QT += gui


TARGET = OpenCV_Template
CONFIG += c++11 console
CONFIG -= app_bundle

TEMPLATE = app
INCLUDEPATH += C:\opencv-3.4.3\release\install\include

LIBS += C:\opencv-3.4.3\release\bin\libopencv_core343.dll
LIBS += C:\opencv-3.4.3\release\bin\libopencv_highgui343.dll
LIBS += C:\opencv-3.4.3\release\bin\libopencv_imgcodecs343.dll
LIBS += C:\opencv-3.4.3\release\bin\libopencv_imgproc343.dll
LIBS += C:\opencv-3.4.3\release\bin\libopencv_calib3d343.dll
LIBS += C:\opencv-3.4.3\release\bin\libopencv_objdetect343.dll
LIBS += C:\opencv-3.4.3\release\bin\libopencv_ml343.dll
LIBS += C:\opencv-3.4.3\release\bin\libopencv_videoio343.dll

SOURCES += \
    HOGImage.cpp \
    main.cpp


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    HOGImage.h

DISTFILES += \
    detekcija_ribe.xml


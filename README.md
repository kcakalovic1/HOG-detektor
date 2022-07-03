# HOG-detektor
Implementacija HOG detektora objekata na proizvoljnim digitalnim slikama pomoću Qt/OpenCV okruženja i biblioteka u programskom jeziku C++.

Sistem za detekciju proizvoljnih objekata iz okruženja korištenjem OpenCV-a i C++ programskog jezika: 
Prilikom implementacije HOG detektora objekata u OpenCV-u korištena je biblioteka opencv2/objdetect.hpp, koja sadrži strukturu cv::HOGDescriptor. 
Detektor je implementiran pomoću SVM klasifikatora i HOG deskriptora. 
Algoritam rada HOG detektora baziran je na prvobitnom algoritmu HOG-a, predstavljenom kroz rad “Histograms of oriented gradients for human detection” (Dalal, N., Triggs, B). 
Testna detekcija bazirana je na detekciji specifičnih dijelova riba (očiju riba) - na slikama 8 testnih klasa riba. 

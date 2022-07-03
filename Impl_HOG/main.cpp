#include <HOGImage.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

void ResizeBoxes(cv::Rect& box) {
    box.x += cvRound(box.width*0.1);
    box.width = cvRound(box.width*0.8);
    box.y += cvRound(box.height*0.06);
    box.height = cvRound(box.height*0.8);
}

vector< float > get_svm_detector( const Ptr< SVM >& svm );
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml( const std::vector< Mat > & train_samples, Mat& trainData );
void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages );
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size );
void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip );
void test_trained_detector( String obj_det_filename, String test_dir, String videofilename );


void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
    // support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
    CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
        (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
    CV_Assert(sv.type() == CV_32F);
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
}

vector< float > get_svm_detector( const Ptr< SVM >& svm )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );

    vector< float > hog_detector( sv.cols + 1 );
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}

void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
{
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );

    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );

        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages = false )
{
    vector< String > files;
    glob( dirname, files );

    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] ); // ucitavanje slike
        if ( img.empty() )
        {
            cout << files[i] << " is invalid!" << endl; // greska prilikom ucitavanja slike -> skip
            continue;
        }

        if ( showImages )
        {
            imshow( "image", img );
            waitKey( 1 );
        }
        img_lst.push_back( img );
    }
}

void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size )
{
    Rect box;
    box.width = size.width;
    box.height = size.height;

    srand( (unsigned int)time( NULL ) );

    for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        if ( full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height )
        {
            box.x = rand() % ( full_neg_lst[i].cols - box.width );
            box.y = rand() % ( full_neg_lst[i].rows - box.height );
            Mat roi = full_neg_lst[i]( box );
            neg_lst.push_back( roi.clone() );
        }
}

void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip )
{
    HOGDescriptor hog;
    hog.winSize = wsize;

    Mat gray;
    vector< float > descriptors;

    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height )
        {
            Rect r = Rect(( img_lst[i].cols - wsize.width ) / 2,
                          ( img_lst[i].rows - wsize.height ) / 2,
                          wsize.width,
                          wsize.height);
            cvtColor( img_lst[i](r), gray, COLOR_BGR2GRAY );
            hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
            gradient_lst.push_back( Mat( descriptors ).clone() );
            if ( use_flip )
            {
                flip( gray, gray, 1 );
                hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
                gradient_lst.push_back( Mat( descriptors ).clone() );
            }
        }
    }
}



void detekcija_video(String obj_det_filename, String test_dir, String videofilename)
{
    cout << "Testiranje detekcije na video snimku..." << endl;
    HOGDescriptor hog;
    hog.load( obj_det_filename );

    vector< String > files;
    glob(test_dir, files);

    int delay = 0;
    VideoCapture cap;

    if (videofilename != "" )
    {
        if ( videofilename.size() == 1 && isdigit(videofilename[0] ))
            cap.open( videofilename[0] - '0' );
        else
            cap.open( videofilename );
    }

    obj_det_filename = "Testiranje detekcije na video snimku ";
    namedWindow(obj_det_filename, WINDOW_NORMAL);

    for(size_t i=0; ; i++)
    {
        Mat img;

        if (cap.isOpened())
        {
            cap >> img;
            delay = 1;
        }
        else if(i < files.size())
        {
            img = imread(files[i]);
        }

        if (img.empty())
        {
            return;
        }

        vector< Rect > detections;
        vector< double > foundWeights;

        hog.detectMultiScale(img, detections, foundWeights);

        for (size_t j = 0; j < detections.size(); j++)
        {
            ResizeBoxes(detections[j]);
            rectangle( img, detections[j], Scalar(100, 200, 0), img.cols/400 + 1);


        }


        imshow(obj_det_filename, img);

        if(waitKey(delay) == 27)
        {
            return;
        }
    }

}


// --------------------------------------------------------------------
//                   Kruzenje uzoraka i resize
// ---------------------------------------------------------------------
void load_and_res_images(const String & dirname, vector<Mat> & img_lst, bool showImages = false)
{
    // oznacavanje objekata -> prilagodjavanje velicine za deskriptor -> spremanje uzoraka
    vector< String > files;
    glob(dirname, files);

    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] ); // ucitavanje slike
        Size nova = img.size();
        nova = Size(nova.width*0.8, nova.height*0.8) ;
        resize(img, img, nova);
        if (img.empty())
        {
            cout << files[i] << " - pogresno ucitano!" << endl; // greska prilikom ucitavanja slike -> skip
            continue;
        }
        // oznacavanje objekta od interesa za kruzenje
        bool fromCenter = false; // sa krajeva (false) ili iz centra (true)
        Rect2d r = selectROI("Oznacite zeljeni objekat", img, fromCenter);
        waitKey(0);
        Mat imCrop = img(r);
        // optimalne dimenzije poz/neg uzoraka: 50 x 50
        resize(imCrop, imCrop, Size(50,50));
        img_lst.push_back(imCrop);
        imwrite(files[i], imCrop);
        if (showImages)
        {
            imshow( "Iskruzeni dio slike/uzorak", imCrop);
            waitKey(0);
        }

    }
}

// --------------------------------------------------------------------
//                        Vizualizacija gradijenata
// ---------------------------------------------------------------------
void vizualizacija_gradijenata()
{
    Mat image = imread("C:/Users/Lenovo/Desktop/testna_slika_1.jpg",1);
    imshow("Testna slika - Vizualizacija gradijenata",image);
    waitKey(0);
    HOGDescriptor hogDesc(image.size(),
                          Size(40, 40),
                          Size(20, 20),
                          Size(20, 20),
                          9);
    Mat hogImg = HOGImage(image, hogDesc, 3, 3,0);

    string name = "Vizualizacija gradijenata";
    namedWindow(name);
    imshow(name, hogImg);
    waitKey(0);
    imwrite("C:/Users/Lenovo/Desktop/hogImg.jpg", hogImg);
    waitKey(0);
    destroyAllWindows();

}

// --------------------------------------------------------------------
//                 Treniranje i Detekcija HOG detektorom
// ---------------------------------------------------------------------

int main() {

        String pos_dir = "...//Slike - HOG detektor - Detekcija ribe//Uzorci za treniranje//Pozitivni_ribe"; // Direktorij sa pozitivnim uzorcima za treniranje
        String neg_dir = "...//Slike - HOG detektor - Detekcija ribe//Uzorci za treniranje//Negativni_ribe"; // Direktorij sa negativnim uzorcima za treniranje
        String obj_det_filename = "...//detekcija_ribe.xml";

        bool visualization = 0;
        bool flip_samples = 0;

        if( pos_dir.empty() || neg_dir.empty() )
        {
            cout << "Navedeni direktoriji uzoraka su prazni.\n\n" << endl;
            exit(1);
        }

        vector< Mat > pos_lst, full_neg_lst, neg_lst, gradient_lst;
        vector< int > labels;

        clog << "Ucitavanje pozitivnih uzoraka ..." ;
        load_images( pos_dir, pos_lst, visualization );
        if ( pos_lst.size() > 0 )
        {
            clog << "...[done] " << pos_lst.size() << " files." << endl;
        }
        else
        {
            clog << "Nema slika za ucitavanje u " << pos_dir <<endl;
            return 1;
        }

        Size pos_image_size = pos_lst[0].size();

        for ( size_t i = 0; i < pos_lst.size(); ++i )
            {
                if( pos_lst[i].size() != pos_image_size )
                {
                    cout << "Svi pozivni uzorci moraju imati iste dimenzije. " << endl;
                    exit( 1 );
                }
            }
            pos_image_size = pos_image_size / 8 * 8;


        clog << "Ucitavanje negativnih uzoraka ...";
        load_images( neg_dir, neg_lst, visualization );
        clog << "...[done] " << neg_lst.size() << " files." << endl;
       full_neg_lst = neg_lst;
        clog << "Procesiranje negativnih uzoraka ...";
        sample_neg( full_neg_lst, neg_lst, pos_image_size);
        clog << "...[done] " << full_neg_lst.size() << " files." << endl;

        clog << "Proracunavanje histograma gradijenata za pozitivne uzorke ... ";
        computeHOGs( pos_image_size, pos_lst, gradient_lst, flip_samples );
        size_t positive_count = gradient_lst.size();
        labels.assign( positive_count, +1 );
        clog << "...[done] ( Prebrojani pozitivni uzorci: " << positive_count << " )" << endl;

        clog << "Proracunavanje histograma gradijenata za negativne uzorke ...";
        computeHOGs( pos_image_size, full_neg_lst, gradient_lst, flip_samples );
        size_t negative_count = gradient_lst.size() - positive_count;
        labels.insert( labels.end(), negative_count, -1 );
        CV_Assert( positive_count < labels.size() );
        clog << "...[done] ( Prebrojani negativni uzroci: " << negative_count << " )" << endl;

        Mat train_data;
        convert_to_ml( gradient_lst, train_data );

        clog << "Treniranje SVM-a ...";
        Ptr< SVM > svm = SVM::create();
        Ptr< SVM > m_svm = SVM::create();

        // Default values za treniranje SVM-a
        svm->setCoef0( 0.0 );
        svm->setDegree(3);
        svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
        svm->setGamma(0);
        svm->setKernel(SVM::LINEAR);
        svm->setNu(0.5);
        svm->setP(0.1);
        svm->setC(0.01); // soft classifier
        svm->setType(SVM::EPS_SVR); // EPS_SVR
        svm->train(train_data, ROW_SAMPLE, labels);


        clog << "...[done]" << endl;


    // DETEKCIJA NA SLICI

        HOGDescriptor hog;
       hog.winSize = pos_image_size;

        hog.setSVMDetector(get_svm_detector(svm));
        hog.save(obj_det_filename); // obj_det_filename - .xml file kreiran treniranjem
        // testiranje detekcije
        vector< Rect > detections;
        vector< double > foundWeights;

        vector <Rect> r_detections;
        vector < double >  r_foundWeights;
        Mat img = imread("...//Slike - HOG detektor - Detekcija ribe//Testne slike//1_.jpg",1); // ucitavanje testne slike nad kojom se vrsi detekcija

        hog.detectMultiScale(img, detections, foundWeights);

        // kontrola ’True Positive’/'False Positive’ detekcija -> prema pouzdanosti (udaljenosti uzoraka od hiperravni SVM-a)
        for (size_t j = 0; j < detections.size(); j++)
        {
           if (foundWeights[j] > 0.6) {
          r_detections.push_back(detections[j]);
          r_foundWeights.push_back(foundWeights[j]);
        }
        }
        cout << "Detektovano: " << r_detections.size() << " objekata" << endl; // broj detektovanih objekata na slici
        for ( size_t j = 0; j < r_detections.size(); j++ )
        {
            if (r_detections.empty()) {
                cout << "Nisu detektovani objekti od interesa." << endl;
                break;
            }
          ResizeBoxes(r_detections[j]); // prilagodjavanje velicine oznake detektovanog objekta na slici
          rectangle( img, r_detections[j], Scalar(100, 200, 0), img.cols/400 + 1); // oznacavanje detektovanog objekta box-om
        }


        bool ispis_lokacija = false;
        int broj = r_detections.size();
        String broj_s = to_string(broj);
        String text = "Detektovano: [";
        text = text + broj_s;
        text = text + "] objekat/a";
        int x,y;
        x = img.cols;
        y = img.rows;
        Point pozicija_teksta(x,y);
        putText(img, text, Point(10,y-10) , FONT_HERSHEY_COMPLEX_SMALL, 3, Scalar(200, 200, 0), 2); // broj detektovanih objekata
        for (uint i = 0; i < r_detections.size(); i++) {

        // Poudanost - skalirana sa 100 radi jednostavnijeg ocitavanja na testnim slikama
        String oznaka = "D" + to_string(i + 1) + ": " + to_string(r_foundWeights[i]*100) + "%";// dodavanje oznake o podudarnosti sa poz uzorkom

        // ispis lokacija: TopLeft i BottomRight tacke box-a detektovanog objekta
        if (ispis_lokacija == true) {
            Point lokacijaTL = r_detections[i].tl();
            Point lokacijaBR = r_detections[i].br();
            cout << "Detekcija " << i + 1 << ". ->  TopLeft: " << lokacijaTL << " |  BottomRight: " << lokacijaBR <<  endl;
            cout << "Pouzdanost: "<< r_foundWeights[i]*100  << "\n" <<endl; // Poudanost - skalirana sa 100 radi jednostavnijeg ocitavanja na testnim slikama
        }
        putText(img, oznaka , r_detections[i].tl() , FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 200, 150), 2);

        }
        imshow("Rezultat detekcije objekta na slici", img);
        waitKey(0);

return 0;
}

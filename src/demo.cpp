#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

int main(int argc, char** argv)
{

   cv::String keys =
        "{o option |<none>           | 1 input is an image, 2 input is a video}" 
        "{i image |           | old image path}"                                           
        "{n new |./resource/image/new_scenery.jpg        | new image path}"                           
        "{v video |           | video path}"                           
        "{t output |./result/image/output.jpg          | output image}"                                                   
        "{q out |./result/video/output.avi           | output video}"                                                                                                            
        "{help h usage ?    |      | show help message}";      
  
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Augmented Reality");
    if (parser.has("help")) 
    {
        parser.printMessage();
        return 0;
    }
    int option = parser.get<int>("option");
    cv::String old = parser.get<cv::String>("image"); 
    cv::String newImage = parser.get<cv::String>("new"); 
    cv::String videoPath = parser.get<cv::String>("video"); 
    cv::String output = parser.get<cv::String>("output"); 
    cv::String out = parser.get<cv::String>("out"); 

 
    if (!parser.check()) 
    {
        parser.printErrors();
        return -1;
    }
    // store image/video to process
    cv::String str, outputFile;

    switch (option)
    {
    case 1:
        if (!parser.has("image")) 
        {
            std::cout<<"Please insert an image path\n";
            return 0;
        }
        str = old;
        outputFile = output;
        break;
    case 2:
        if (!parser.has("video")) 
        {
            std::cout<<"Please insert an video path\n";
            return 0;
            str = videoPath;
        }
        outputFile = out;
        break;
    default:
        std::cout <<"Please insert a choice again.\n";
        break;
    }

    // Open a video file or an image file or a camera stream.
    //string str, outputFile;
    cv::VideoCapture cap;
    cv::VideoWriter video;
    cv::Mat frame, blob;
    
    //cv::Mat im_src = imread("new_scenery.jpg");
    cv::Mat im_src = cv::imread(newImage);

    // Get the video writer initialized to save the output video
    cap.open(str);
    if (!parser.has("image")) 
    {
        video.open(outputFile, cv::VideoWriter::fourcc('M','J','P','G'), 
        28, cv::Size(2*cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    }
    
    // Create a window
    static const std::string kWinName = "Augmented Reality using Aruco markers in OpenCV";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

    // Process frames.
    while (cv::waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;
        
        try {
            // Stop the program if reached end of video
            if (frame.empty()) 
            {
                std::cout << "Done processing !!!\n";
                std::cout << "Output file is stored as " << outputFile << "\n";
                cv::waitKey(3000);
                break;
            }

            std::vector<int> markerIds;
            
            // Load the dictionary that was used to generate the markers.
            cv::Ptr<cv::aruco::Dictionary>  dictionary = cv::aruco::getPredefinedDictionary( cv::aruco::DICT_6X6_250);

            // Declare the vectors that would contain the detected marker corners and the rejected marker candidates
            std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

            // Initialize the detector parameters using default values
            cv::Ptr<cv::aruco::DetectorParameters> parameters =  cv::aruco::DetectorParameters::create();

            // Detect the markers in the image
            cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

            // Using the detected markers, locate the quadrilateral on the target frame where the new scene is going to be displayed.
            std::vector<cv::Point> pts_dst;
            float scalingFac = 0.02;//0.015;

            cv::Point refPt1, refPt2, refPt3, refPt4;

            // finding top left corner point of the target quadrilateral
            std::vector<int>::iterator it = std::find(markerIds.begin(), markerIds.end(), 25);
            int index = std::distance(markerIds.begin(), it);
            refPt1 = markerCorners.at(index).at(1);

            // finding top right corner point of the target quadrilateral
            it = std::find(markerIds.begin(), markerIds.end(), 33);
            index = std::distance(markerIds.begin(), it);
            refPt2 = markerCorners.at(index).at(2);
            
            float distance = norm(refPt1-refPt2);
            pts_dst.push_back(cv::Point(refPt1.x - round(scalingFac*distance), refPt1.y - round(scalingFac*distance)));
            
            pts_dst.push_back(cv::Point(refPt2.x + round(scalingFac*distance), refPt2.y - round(scalingFac*distance)));

            // finding bottom right corner point of the target quadrilateral
            it = std::find(markerIds.begin(), markerIds.end(), 30);
            index = std::distance(markerIds.begin(), it);
            refPt3 = markerCorners.at(index).at(0);
            pts_dst.push_back(cv::Point(refPt3.x + round(scalingFac*distance), refPt3.y + round(scalingFac*distance)));

            // finding bottom left corner point of the target quadrilateral
            it = std::find(markerIds.begin(), markerIds.end(), 23);
            index = std::distance(markerIds.begin(), it);
            refPt4 = markerCorners.at(index).at(0);
            pts_dst.push_back(cv::Point(refPt4.x - round(scalingFac*distance), refPt4.y + round(scalingFac*distance)));

            // Get the corner points of the new scene image.
            std::vector<cv::Point> pts_src;
            pts_src.push_back(cv::Point(0,0));
            pts_src.push_back(cv::Point(im_src.cols, 0));
            pts_src.push_back(cv::Point(im_src.cols, im_src.rows));
            pts_src.push_back(cv::Point(0, im_src.rows));

            // Compute homography from source and destination points
            cv::Mat h = cv::findHomography(pts_src, pts_dst);

            // Warped image
            cv::Mat warpedImage;
            
            // Warp source image to destination based on homography
            cv::warpPerspective(im_src, warpedImage, h, frame.size(), cv::INTER_CUBIC);
        
            // Prepare a mask representing region to copy from the warped image into the original frame.
            cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
            cv::fillConvexPoly(mask, pts_dst, cv::Scalar(255, 255, 255), cv::LINE_AA);
            
            // Erode the mask to not copy the boundary effects from the warping
            cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(5,5));
//            cv::Mat element = cv::getStructuringElement( MORPH_RECT, cv::Size(3,3));
            erode(mask, mask, element);

            // Copy the warped image into the original frame in the mask region.
            cv::Mat imOut = frame.clone();
            warpedImage.copyTo(imOut, mask);
            
            // Showing the original image and the new output image side by side
            cv::Mat concatenatedOutput;
            cv::hconcat(frame, imOut, concatenatedOutput);
            //imshow("zouma", concatenatedOutput);
            cv::waitKey(0);
            if (parser.has("image")) cv::imwrite(outputFile, concatenatedOutput);
            else video.write(concatenatedOutput);

            cv::imshow(kWinName, concatenatedOutput);
            
        }
        catch(const std::exception& e) 
        {
            std::cout << "\n e : " << e.what() << "\n";
            std::cout << "Could not do homography !! \n";
    //        return 0;
        }

    }

    cap.release();
    if (!parser.has("image")) video.release();

    return 0;
}

// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how you can use dlib to make an object detector
    for things like faces, pedestrians, and any other semi-rigid object.  In
    particular, we go though the steps to train the kind of sliding window
    object detector first published by Dalal and Triggs in 2005 in the paper
    Histograms of Oriented Gradients for Human Detection.

    Note that this program executes fastest when compiled with at least SSE2
    instructions enabled.  So if you are using a PC with an Intel or AMD chip
    then you should enable at least SSE2 instructions.  If you are using cmake
    to compile this program you can enable them by using one of the following
    commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.

*/


#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <iostream>
#include <fstream>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------


int main(int argc, char** argv){

    try
    {
        if (argc != 3)
        {
            cout << "Give the path to the examples/faces directory as the argument to this" << endl;
            cout << "program.  For example, if you are in the examples folder then execute " << endl;
            cout << "this program by running: " << endl;
            cout << "   ./fhog_object_detector_ex faces" << endl;
            cout << endl;
            return 0;
        }
        const std::string faces_file = argv[1];
        int size = atoi(argv[2]);
        dlib::array<array2d<unsigned char> > images_train;
        std::vector<std::vector<rectangle> > face_boxes_train;
        load_image_dataset(images_train, face_boxes_train, faces_file);
        upsample_image_dataset<pyramid_down<2> >(images_train, face_boxes_train);
        cout << "num training images: " << images_train.size() << endl;
        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
        image_scanner_type scanner;
        scanner.set_detection_window_size(size, size);
        structural_object_detection_trainer<image_scanner_type> trainer(scanner);
        trainer.set_num_threads(4);
        trainer.set_c(1);
        trainer.be_verbose();
        trainer.set_epsilon(0.01);
        object_detector<image_scanner_type> detector = trainer.train(images_train, face_boxes_train);
        cout << "training results: " << test_object_detection_function(detector, images_train, face_boxes_train) << endl;
        image_window hogwin(draw_fhog(detector), "Learned fHOG detector");
        serialize("face_detector.svm") << detector;
        // Then you can recall it using the deserialize() function.
        //object_detector<image_scanner_type> detector2;
        //deserialize("face_detector.svm") >> detector2;
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------




//////////////////////////////////////////////////////////////////////////
// Color Transfer between Images, E. Reinhard et al, 2001
// Implemented by Jun Yan, 2013/12/09

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>

using namespace cv;
using namespace std;

Mat rgb2lms;
Mat loglms2lab;

float M_rgb2lms[]={
	0.3811f, 0.5783f, 0.0402f,
	0.1967f, 0.7244f, 0.0782f,
	0.0241f, 0.1228f, 0.8444f
};

float M_loglms2lab_l[]={
	1.f/sqrtf(3.f), 0.f, 0.f,
	0.f, 1.f/sqrtf(6.f), 0.f,
	0.f, 0.f, 1.f/sqrtf(2.f),
};

float M_loglms2lab_r[]={
	1.f,1.f,1.f,
	1.f,1.f,-2.f,
	1.f,-1.f,0.f
};

//////////////////////////////////////////////////////////////////////////
// inverse
float M_lms2rgb[]={
	4.4679f, -3.5873f, 0.1193f,
	-1.2186f, 2.3809f, -0.1624f,
	0.0497f, -0.2439f, 1.2045f
};

// same with M_loglms2lab_l??
float M_lab2loglms_r[]={
	1.f/sqrtf(3.f), 0.f, 0.f,
	0.f, 1.f/sqrtf(6.f), 0.f,
	0.f, 0.f, 1.f/sqrtf(2.f),
};

float M_lab2loglms_l[]={
	1.f,1.f,1.f,
	1.f,1.f,-1.f,
	1.f,-2.f,0.f
};

void rgb2lab(const Mat& rgb,Mat* lab)
{
	Mat tr_rgb2lms(3,3,CV_32FC1,M_rgb2lms);
	Mat tr_loglms2lab_l(3,3,CV_32FC1,M_loglms2lab_l);
	Mat tr_loglms2lab_r(3,3,CV_32FC1,M_loglms2lab_r);

	Mat lms;
	cv::transform(rgb,lms,tr_rgb2lms);
	// log10(x)=ln(x)/ln(10)
	cv::log(lms,lms);
	float inv_ln10=1.f/logf(10.f);
	lms*=inv_ln10;

	cv::transform(lms,*lab,tr_loglms2lab_l*tr_loglms2lab_r);
}

void lab2rgb(const Mat& lab,Mat* rgb)
{
	Mat tr_lms2rgb(3,3,CV_32FC1,M_lms2rgb);
	Mat M_lab2loglms_l(3,3,CV_32FC1,M_lab2loglms_l);
	Mat M_lab2loglms_r(3,3,CV_32FC1,M_lab2loglms_r);

	Mat lms;
	cv::transform(lab,lms,M_lab2loglms_l*M_lab2loglms_r);
	// 10^x=(e^x)^(ln10)
	cv::exp(lms,lms);
	float ln10=logf(10.f);
	cv::pow(lms,ln10,lms);

	cv::transform(lms,*rgb,tr_lms2rgb);
}


#define USE_MY_CVT


int main(int argc, char** argv)
{
	char* file1,*file2;
	if(argc!=3)
	{
		printf("Usage: colortransfer file1 file2\n");

		printf("No input files specified, trying to load 1.png and 2.png\n");
		file1="1.png";
		file2="2.png";
	}
	else
	{
		file1=argv[1];
		file2=argv[2];
	}

	Mat img_src=imread(file1);
	Mat img_targ=imread(file2);
	imshow("Source",img_src);
	imshow("Target",img_targ);

	Mat lab_img_src,lab_img_targ;
#ifdef USE_MY_CVT
	// we need rgb order!! and convert to 0-1 range
	cv::cvtColor(img_src, img_src, CV_BGR2RGB);
	cv::cvtColor(img_targ, img_targ, CV_BGR2RGB);
	img_src.convertTo(img_src, CV_32FC3,1.f/255.f);  
	img_targ.convertTo(img_targ, CV_32FC3, 1.f/255.0f);	
	
	rgb2lab(img_src,&lab_img_src);
	rgb2lab(img_targ,&lab_img_targ);
#else
	cvtColor(img_src, lab_img_src, CV_BGR2Lab);
	cvtColor(img_targ, lab_img_targ, CV_BGR2Lab);
#endif


 	Scalar m_src,stddev_src,m_targ,stddev_targ;
	meanStdDev(lab_img_src,m_src,stddev_src);
	meanStdDev(lab_img_targ,m_targ,stddev_targ);
	
	// todo: make it an affine transform?
	Mat result=lab_img_src-m_src;
	vector<Mat> channels(3);
	split(result, channels);
	channels[0]*=stddev_targ[0]/stddev_src[0];
	channels[1]*=stddev_targ[1]/stddev_src[1];
	channels[2]*=stddev_targ[2]/stddev_src[2];
	merge(channels,result);
	result+=m_targ;

#ifdef USE_MY_CVT
	lab2rgb(result,&result);
	cv::cvtColor(result, result, CV_RGB2BGR);
#else
	cvtColor(result, result, CV_Lab2BGR);
#endif


	imshow("Result",result);

	// convert to 24 bits before save
	result.convertTo(result, CV_8UC3,255.f);  
	imwrite("result.png",result);
	printf("Result saved in 'result.png'\n");

	waitKey(0);

	return 0;
}



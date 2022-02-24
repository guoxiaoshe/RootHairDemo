#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <string>
#include <sstream>

using namespace cv;

int grayBin();
int repair();
int deleaf();
int deLone();
void delete_jut();
void imageblur();
void HilditchThin1();
int myThin();
int measure();

//定义全局的原图形矩阵
Mat srcmat;//Mat &src = srcmat;
Mat srcmatGray;//Mat &srcG = srcmatGray;//读取图像并进行灰度化
Mat dismat1;//Mat &dis1 = dismat1;//储存二值化结果
Mat dismat2;//储存降噪结果
Mat dismat3;//储存去除叶子的结果
Mat dismat4;//存储去除孤立小区域的结果
Mat dismat5;//存储骨架
Mat result;
Mat tmp;//测试用

int main() {
	//定义全局的原图形矩阵
	srcmat = imread("test03.jpg");
	srcmatGray = imread("test03.jpg", 0);//读取图像并进行灰度化
	
	grayBin();//二值化
	repair();//图像降噪
	deleaf();//去除叶子
	deLone();//区域删除
	myThin();//细化
	measure();//测量
	return 0;
}

//后续：定义全局变量存储文件路径


//二值化
//后续添加用户可调的二值化窗口
int grayBin() {

	threshold(srcmatGray, dismat1, 85, 255, THRESH_BINARY);//后续可优化使用大津法
	imshow("original", srcmat);
	//imshow("bin", dismat1);
	//waitKey(0);
	imwrite("binresult.jpg", dismat1);

	return 0;
}

//图像修复与降噪
int repair() {
	//闭合运算连接根毛
	Mat element3(3, 3, CV_8U, Scalar(1));
	Mat closed;
	morphologyEx(dismat1, closed, MORPH_CLOSE, element3);
	//imshow("repaired", closed);

	//开运算降噪
	Mat element4(4, 4, CV_8U, Scalar(1));
	Mat opened;
	morphologyEx(closed, opened, MORPH_OPEN, element4);
	//imshow("denoised", opened);
	//waitKey(0);//used foe debug
	opened.copyTo(dismat2);
	imwrite("denoised.jpg", opened);

	return 0;
}

//去除叶子
int deleaf() {
	//腐蚀根
	Mat element13(13, 13, CV_8U, Scalar(1));
	Mat eroded;
	erode(dismat2, eroded, element13);
	//imshow("erodedRoot", eroded);used for debug

	//膨胀叶子
	Mat dilated;
	dilate(eroded, dilated, element13);

	//与原图相减并再次降噪
	Mat element4(4, 4, CV_8U, Scalar(1));
	Mat subed;
	subtract(dismat2, dilated, subed);
	//imshow("aubtract", subed);
	morphologyEx(subed, dismat3, MORPH_OPEN, element4);
	//imshow("deleaf", dismat3);
	
	//waitKey(0);
	imwrite("deleaf.jpg", dismat3);
	
	return 0;
}

//去除孤立区域
int deLone() {
	//提取轮廓
	std::vector<std::vector<cv::Point>> contours;//存轮廓的向量
	findContours(dismat3, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//在黑色图像上画出白色轮廓
	Mat profile(srcmat.size(), CV_8U, Scalar(225));
	drawContours(profile, contours,
		-1,//画出全部
		0,//黑色
		2);//宽度为2
	//imshow("profile", profile);

	//删除面积小的区域
	//轮廓长度策略
	dismat3.copyTo(dismat4);
	double area = 0.0;
	for (int i = 0;i < contours.size();i++) {
		area = contourArea(contours[i]);
		if (area < 770)
			drawContours(dismat4, contours,
				i,//第i个区域
				0,//填充颜色
				-1);//按照填充方式处理
	}
	//imshow("deLone", dismat4);
	//waitKey(0);
	imwrite("deLone.jpg", dismat4);

	return 0;
}

//去除边缘突出部
//去除二值图像边缘的突出部
//uthreshold、vthreshold分别表示突出部的宽度阈值和高度阈值
//type代表突出部的颜色，0表示黑色，1代表白色 
void delete_jut(Mat& src, Mat& dst, int uthreshold, int vthreshold, int type)
{
	int threshold;
	src.copyTo(dst);
	int height = dst.rows;
	int width = dst.cols;
	int k;  //用于循环计数传递到外部
	for (int i = 0; i < height - 1; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < width - 1; j++)
		{
			if (type == 0)
			{
				//行消除
				if (p[j] == 255 && p[j + 1] == 0)
				{
					if (j + uthreshold >= width)
					{
						for (int k = j + 1; k < width; k++)
							p[k] = 255;
					}
					else
					{
						for (k = j + 2; k <= j + uthreshold; k++)
						{
							if (p[k] == 255) break;
						}
						if (p[k] == 255)
						{
							for (int h = j + 1; h < k; h++)
								p[h] = 255;
						}
					}
				}
				//列消除
				if (p[j] == 255 && p[j + width] == 0)
				{
					if (i + vthreshold >= height)
					{
						for (k = j + width; k < j + (height - i)*width; k += width)
							p[k] = 255;
					}
					else
					{
						for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
						{
							if (p[k] == 255) break;
						}
						if (p[k] == 255)
						{
							for (int h = j + width; h < k; h += width)
								p[h] = 255;
						}
					}
				}
			}
			else  //type = 1
			{
				//行消除
				if (p[j] == 0 && p[j + 1] == 255)
				{
					if (j + uthreshold >= width)
					{
						for (int k = j + 1; k < width; k++)
							p[k] = 0;
					}
					else
					{
						for (k = j + 2; k <= j + uthreshold; k++)
						{
							if (p[k] == 0) break;
						}
						if (p[k] == 0)
						{
							for (int h = j + 1; h < k; h++)
								p[h] = 0;
						}
					}
				}
				//列消除
				if (p[j] == 0 && p[j + width] == 255)
				{
					if (i + vthreshold >= height)
					{
						for (k = j + width; k < j + (height - i)*width; k += width)
							p[k] = 0;
					}
					else
					{
						for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
						{
							if (p[k] == 0) break;
						}
						if (p[k] == 0)
						{
							for (int h = j + width; h < k; h += width)
								p[h] = 0;
						}
					}
				}
			}
		}
	}
}

//图片边缘光滑处理
//size表示取均值的窗口大小，threshold表示对均值图像进行二值化的阈值
void imageblur(Mat& src, Mat& dst, Size size, int threshold)
{
	int height = src.rows;
	int width = src.cols;
	blur(src, dst, size);
	for (int i = 0; i < height; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
		{
			if (p[j] < threshold)
				p[j] = 0;
			else p[j] = 255;
		}
	}
	imshow("Blur", dst);
}


//细化函数
void HilditchThin1(Mat &src, Mat &dst)
{
	//http://cgm.cs.mcgill.ca/~godfried/teaching/projects97/azar/skeleton.html#algorithm
	//算法有问题，得不到想要的效果
	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	int i, j;
	int width, height;
	//之所以减2，是方便处理8邻域，防止越界
	width = src.cols - 2;
	height = src.rows - 2;
	int step = src.step;
	int  p2, p3, p4, p5, p6, p7, p8, p9;
	uchar* img;
	bool ifEnd;
	int A1;
	cv::Mat tmpimg;
	while (1)
	{
		dst.copyTo(tmpimg);
		ifEnd = false;
		img = tmpimg.data + step;
		for (i = 2; i < height; i++)
		{
			img += step;
			for (j = 2; j<width; j++)
			{
				uchar* p = img + j;
				A1 = 0;
				if (p[0] > 0)
				{
					if (p[-step] == 0 && p[-step + 1]>0) //p2,p3 01模式
					{
						A1++;
					}
					if (p[-step + 1] == 0 && p[1]>0) //p3,p4 01模式
					{
						A1++;
					}
					if (p[1] == 0 && p[step + 1]>0) //p4,p5 01模式
					{
						A1++;
					}
					if (p[step + 1] == 0 && p[step]>0) //p5,p6 01模式
					{
						A1++;
					}
					if (p[step] == 0 && p[step - 1]>0) //p6,p7 01模式
					{
						A1++;
					}
					if (p[step - 1] == 0 && p[-1]>0) //p7,p8 01模式
					{
						A1++;
					}
					if (p[-1] == 0 && p[-step - 1]>0) //p8,p9 01模式
					{
						A1++;
					}
					if (p[-step - 1] == 0 && p[-step]>0) //p9,p2 01模式
					{
						A1++;
					}
					p2 = p[-step]>0 ? 1 : 0;
					p3 = p[-step + 1]>0 ? 1 : 0;
					p4 = p[1]>0 ? 1 : 0;
					p5 = p[step + 1]>0 ? 1 : 0;
					p6 = p[step]>0 ? 1 : 0;
					p7 = p[step - 1]>0 ? 1 : 0;
					p8 = p[-1]>0 ? 1 : 0;
					p9 = p[-step - 1]>0 ? 1 : 0;
					//计算AP2,AP4
					int A2, A4;
					A2 = 0;
					//if(p[-step]>0)
					{
						if (p[-2 * step] == 0 && p[-2 * step + 1]>0) A2++;
						if (p[-2 * step + 1] == 0 && p[-step + 1]>0) A2++;
						if (p[-step + 1] == 0 && p[1]>0) A2++;
						if (p[1] == 0 && p[0]>0) A2++;
						if (p[0] == 0 && p[-1]>0) A2++;
						if (p[-1] == 0 && p[-step - 1]>0) A2++;
						if (p[-step - 1] == 0 && p[-2 * step - 1]>0) A2++;
						if (p[-2 * step - 1] == 0 && p[-2 * step]>0) A2++;
					}


					A4 = 0;
					//if(p[1]>0)
					{
						if (p[-step + 1] == 0 && p[-step + 2]>0) A4++;
						if (p[-step + 2] == 0 && p[2]>0) A4++;
						if (p[2] == 0 && p[step + 2]>0) A4++;
						if (p[step + 2] == 0 && p[step + 1]>0) A4++;
						if (p[step + 1] == 0 && p[step]>0) A4++;
						if (p[step] == 0 && p[0]>0) A4++;
						if (p[0] == 0 && p[-step]>0) A4++;
						if (p[-step] == 0 && p[-step + 1]>0) A4++;
					}


					//printf("p2=%d p3=%d p4=%d p5=%d p6=%d p7=%d p8=%d p9=%d\n", p2, p3, p4, p5, p6,p7, p8, p9);
					//printf("A1=%d A2=%d A4=%d\n", A1, A2, A4);
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)>1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)<7 && A1 == 1)
					{
						if (((p2 == 0 || p4 == 0 || p8 == 0) || A2 != 1) && ((p2 == 0 || p4 == 0 || p6 == 0) || A4 != 1))
						{
							dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
							ifEnd = true;
							//printf("\n");

							//PrintMat(dst);
						}
					}
				}
			}
		}
		//printf("\n");
		//PrintMat(dst);
		//PrintMat(dst);
		//已经没有可以细化的像素了，则退出迭代
		if (!ifEnd) break;
	}
}

//细化
int myThin() {
	Mat del;
	delete_jut(dismat4, del, 3, 3, 1);
	//imshow("debump", del);
	/*开运算降噪(作用不明显)
	Mat element4(5, 5, CV_8U, Scalar(1));
	Mat opened;
	morphologyEx(del, opened, MORPH_OPEN, element4);
	imshow("denoised", opened);*/
	/*Mat blured;//效果不明显
	CvSize size = cvSize(2, 2);
	imageblur(del,blured,size,1);*/
	HilditchThin1(dismat4, dismat5);

	/*Point P1, P2;
	P1.x = 1200;
	P1.y = 20;
	P2.x = 1200;
	P2.y = 220;
	cvDrawLine(&dismat5, P1, P2, Scalar(1), 1);*/

	//imshow("thin", dismat5);
	
	//waitKey(0);
	imwrite("thin.jpg", dismat5);


	return 0;
}

//测量
int measure() {
	//提取轮廓
	std::vector<std::vector<cv::Point>> contours;//存轮廓的向量
	findContours(dismat5, contours, 
		RETR_EXTERNAL,//检索外部轮廓
		CHAIN_APPROX_NONE);//每个轮廓的全部像素
	

	//在原图标出轮廓
	srcmat.copyTo(result);
	drawContours(result, contours,
		-1,//画出全部
		Scalar(0,0,255),//red
		2);//宽度为2
	//imshow("redline",result);

	//循环测量
	int size = contours.size();
	const int MAXSIZE = 100;
	double length[MAXSIZE];
	Point numPoint;//表数字的点
	String str;
	for (int i = 0;i < size;i++) {
		length[i] = arcLength(contours[i],0);
		numPoint = contours[i][0];
		std::stringstream ss;
		ss << i;
		str = ss.str();
		putText(result, str, numPoint, FONT_HERSHEY_COMPLEX, 2.0, Scalar(0, 0, 255), 2);
	}
		
	imshow("result", result);
	
	for (int i = 0;i < size;i++) {
		std::cout << "test point" << i << "  " << "length:" << length[i] << std::endl;
	}
	waitKey(0);
	imwrite("result.jpg", result);
	
	return 0;
}
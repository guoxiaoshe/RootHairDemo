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

//����ȫ�ֵ�ԭͼ�ξ���
Mat srcmat;//Mat &src = srcmat;
Mat srcmatGray;//Mat &srcG = srcmatGray;//��ȡͼ�񲢽��лҶȻ�
Mat dismat1;//Mat &dis1 = dismat1;//�����ֵ�����
Mat dismat2;//���潵����
Mat dismat3;//����ȥ��Ҷ�ӵĽ��
Mat dismat4;//�洢ȥ������С����Ľ��
Mat dismat5;//�洢�Ǽ�
Mat result;
Mat tmp;//������

int main() {
	//����ȫ�ֵ�ԭͼ�ξ���
	srcmat = imread("test03.jpg");
	srcmatGray = imread("test03.jpg", 0);//��ȡͼ�񲢽��лҶȻ�
	
	grayBin();//��ֵ��
	repair();//ͼ����
	deleaf();//ȥ��Ҷ��
	deLone();//����ɾ��
	myThin();//ϸ��
	measure();//����
	return 0;
}

//����������ȫ�ֱ����洢�ļ�·��


//��ֵ��
//��������û��ɵ��Ķ�ֵ������
int grayBin() {

	threshold(srcmatGray, dismat1, 85, 255, THRESH_BINARY);//�������Ż�ʹ�ô��
	imshow("original", srcmat);
	//imshow("bin", dismat1);
	//waitKey(0);
	imwrite("binresult.jpg", dismat1);

	return 0;
}

//ͼ���޸��뽵��
int repair() {
	//�պ��������Ӹ�ë
	Mat element3(3, 3, CV_8U, Scalar(1));
	Mat closed;
	morphologyEx(dismat1, closed, MORPH_CLOSE, element3);
	//imshow("repaired", closed);

	//�����㽵��
	Mat element4(4, 4, CV_8U, Scalar(1));
	Mat opened;
	morphologyEx(closed, opened, MORPH_OPEN, element4);
	//imshow("denoised", opened);
	//waitKey(0);//used foe debug
	opened.copyTo(dismat2);
	imwrite("denoised.jpg", opened);

	return 0;
}

//ȥ��Ҷ��
int deleaf() {
	//��ʴ��
	Mat element13(13, 13, CV_8U, Scalar(1));
	Mat eroded;
	erode(dismat2, eroded, element13);
	//imshow("erodedRoot", eroded);used for debug

	//����Ҷ��
	Mat dilated;
	dilate(eroded, dilated, element13);

	//��ԭͼ������ٴν���
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

//ȥ����������
int deLone() {
	//��ȡ����
	std::vector<std::vector<cv::Point>> contours;//������������
	findContours(dismat3, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//�ں�ɫͼ���ϻ�����ɫ����
	Mat profile(srcmat.size(), CV_8U, Scalar(225));
	drawContours(profile, contours,
		-1,//����ȫ��
		0,//��ɫ
		2);//���Ϊ2
	//imshow("profile", profile);

	//ɾ�����С������
	//�������Ȳ���
	dismat3.copyTo(dismat4);
	double area = 0.0;
	for (int i = 0;i < contours.size();i++) {
		area = contourArea(contours[i]);
		if (area < 770)
			drawContours(dismat4, contours,
				i,//��i������
				0,//�����ɫ
				-1);//������䷽ʽ����
	}
	//imshow("deLone", dismat4);
	//waitKey(0);
	imwrite("deLone.jpg", dismat4);

	return 0;
}

//ȥ����Եͻ����
//ȥ����ֵͼ���Ե��ͻ����
//uthreshold��vthreshold�ֱ��ʾͻ�����Ŀ����ֵ�͸߶���ֵ
//type����ͻ��������ɫ��0��ʾ��ɫ��1�����ɫ 
void delete_jut(Mat& src, Mat& dst, int uthreshold, int vthreshold, int type)
{
	int threshold;
	src.copyTo(dst);
	int height = dst.rows;
	int width = dst.cols;
	int k;  //����ѭ���������ݵ��ⲿ
	for (int i = 0; i < height - 1; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < width - 1; j++)
		{
			if (type == 0)
			{
				//������
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
				//������
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
				//������
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
				//������
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

//ͼƬ��Ե�⻬����
//size��ʾȡ��ֵ�Ĵ��ڴ�С��threshold��ʾ�Ծ�ֵͼ����ж�ֵ������ֵ
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


//ϸ������
void HilditchThin1(Mat &src, Mat &dst)
{
	//http://cgm.cs.mcgill.ca/~godfried/teaching/projects97/azar/skeleton.html#algorithm
	//�㷨�����⣬�ò�����Ҫ��Ч��
	if (src.type() != CV_8UC1)
	{
		printf("ֻ�ܴ����ֵ��Ҷ�ͼ��\n");
		return;
	}
	//��ԭ�ز���ʱ��copy src��dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	int i, j;
	int width, height;
	//֮���Լ�2���Ƿ��㴦��8���򣬷�ֹԽ��
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
					if (p[-step] == 0 && p[-step + 1]>0) //p2,p3 01ģʽ
					{
						A1++;
					}
					if (p[-step + 1] == 0 && p[1]>0) //p3,p4 01ģʽ
					{
						A1++;
					}
					if (p[1] == 0 && p[step + 1]>0) //p4,p5 01ģʽ
					{
						A1++;
					}
					if (p[step + 1] == 0 && p[step]>0) //p5,p6 01ģʽ
					{
						A1++;
					}
					if (p[step] == 0 && p[step - 1]>0) //p6,p7 01ģʽ
					{
						A1++;
					}
					if (p[step - 1] == 0 && p[-1]>0) //p7,p8 01ģʽ
					{
						A1++;
					}
					if (p[-1] == 0 && p[-step - 1]>0) //p8,p9 01ģʽ
					{
						A1++;
					}
					if (p[-step - 1] == 0 && p[-step]>0) //p9,p2 01ģʽ
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
					//����AP2,AP4
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
							dst.at<uchar>(i, j) = 0; //����ɾ�����������õ�ǰ����Ϊ0
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
		//�Ѿ�û�п���ϸ���������ˣ����˳�����
		if (!ifEnd) break;
	}
}

//ϸ��
int myThin() {
	Mat del;
	delete_jut(dismat4, del, 3, 3, 1);
	//imshow("debump", del);
	/*�����㽵��(���ò�����)
	Mat element4(5, 5, CV_8U, Scalar(1));
	Mat opened;
	morphologyEx(del, opened, MORPH_OPEN, element4);
	imshow("denoised", opened);*/
	/*Mat blured;//Ч��������
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

//����
int measure() {
	//��ȡ����
	std::vector<std::vector<cv::Point>> contours;//������������
	findContours(dismat5, contours, 
		RETR_EXTERNAL,//�����ⲿ����
		CHAIN_APPROX_NONE);//ÿ��������ȫ������
	

	//��ԭͼ�������
	srcmat.copyTo(result);
	drawContours(result, contours,
		-1,//����ȫ��
		Scalar(0,0,255),//red
		2);//���Ϊ2
	//imshow("redline",result);

	//ѭ������
	int size = contours.size();
	const int MAXSIZE = 100;
	double length[MAXSIZE];
	Point numPoint;//�����ֵĵ�
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
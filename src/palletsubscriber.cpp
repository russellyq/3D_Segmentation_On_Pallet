#include "ros/ros.h"
#include <palletunloader/pallet_report_msg.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

void PalletReportCallback(const palletunloader::pallet_report_msg::ConstPtr& msg)
{
  ROS_INFO("Pallet unloder found pallets as follows:"
           "\n1- [x = %6.3f, y = %6.3f, Theta = %6.3f]"
           "\n2- [x = %6.3f, y = %6.3f, Theta = %6.3f]"
           "\n3- [x = %6.3f, y = %6.3f, Theta = %6.3f]"
           "\n4- [x = %6.3f, y = %6.3f, Theta = %6.3f]",
           msg->Pallet1.position.x,msg->Pallet1.position.y, msg->Pallet1.position.z,
           msg->Pallet2.position.x,msg->Pallet2.position.y, msg->Pallet2.position.z,
           msg->Pallet3.position.x,msg->Pallet3.position.y, msg->Pallet3.position.z,
           msg->Pallet4.position.x,msg->Pallet4.position.y, msg->Pallet4.position.z);
  cv::namedWindow("view");
  cv::startWindowThread();
  try
  {
    cv::Mat image = cv::imdecode(cv::Mat(msg->Camera_Image.data),1);//convert compressed image data to cv::Mat
    cv::imshow("view", image);
    cv::waitKey(10);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert to image!");
  }

}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "palletsubscriber");

  ros::NodeHandle n;
  ros::Subscriber palletsub = n.subscribe<palletunloader::pallet_report_msg>("/PalletReportMsg", 1, &PalletReportCallback);

  ros::spin();
  cv::destroyWindow("view");
  return 0;
}

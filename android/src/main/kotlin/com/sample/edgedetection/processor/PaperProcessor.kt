package com.sample.edgedetection.processor

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import java.time.LocalDateTime
import java.time.temporal.ChronoUnit

const val TAG: String = "PaperProcessor"

var areaHistory : MutableList<Double> = mutableListOf()
lateinit var lastDetectDate : LocalDateTime
lateinit var lastDetectedPoint : List<Point>
 var deviceWidth=0


fun processPicture(previewFrame: Mat): Corners? {
    val contours = findContours(previewFrame)
    return getCorners(contours, previewFrame.size())
}

fun cropPicture(picture: Mat, pts: List<Point>): Mat {

    pts.forEach { Log.i(TAG, "point: $it") }
    val tl = pts[0]
    val tr = pts[1]
    val br = pts[2]
    val bl = pts[3]

    val widthA = sqrt((br.x - bl.x).pow(2.0) + (br.y - bl.y).pow(2.0))
    val widthB = sqrt((tr.x - tl.x).pow(2.0) + (tr.y - tl.y).pow(2.0))

    val dw = max(widthA, widthB)
    val maxWidth = java.lang.Double.valueOf(dw).toInt()


    val heightA = sqrt((tr.x - br.x).pow(2.0) + (tr.y - br.y).pow(2.0))
    val heightB = sqrt((tl.x - bl.x).pow(2.0) + (tl.y - bl.y).pow(2.0))

    val dh = max(heightA, heightB)
    val maxHeight = java.lang.Double.valueOf(dh).toInt()

    val croppedPic = Mat(maxHeight, maxWidth, CvType.CV_8UC4)

    val srcMat = Mat(4, 1, CvType.CV_32FC2)
    val dstMat = Mat(4, 1, CvType.CV_32FC2)

    srcMat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y)
    dstMat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh)

    val m = Imgproc.getPerspectiveTransform(srcMat, dstMat)

    Imgproc.warpPerspective(picture, croppedPic, m, croppedPic.size())
    m.release()
    srcMat.release()
    dstMat.release()
    Log.i(TAG, "crop finish")


    if(croppedPic.height()>1500){
return croppedPic
    }

   else{
    Core.flip(croppedPic, croppedPic,1)
if((croppedPic.width() / croppedPic.height())<0.5){
      Core.rotate(croppedPic, croppedPic, Core.ROTATE_90_COUNTERCLOCKWISE)
        
    }
   } 
    


    return croppedPic
}

fun enhancePicture(src: Bitmap?): Bitmap {
    val srcMat = Mat()
    Utils.bitmapToMat(src, srcMat)
    Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_RGBA2GRAY)
    Imgproc.adaptiveThreshold(
            srcMat,
            srcMat,
            255.0,
            Imgproc.ADAPTIVE_THRESH_MEAN_C,
            Imgproc.THRESH_BINARY,
            15,
            15.0
    )
    val result = Bitmap.createBitmap(src?.width ?: 1080, src?.height ?: 1920, Bitmap.Config.RGB_565)
    Utils.matToBitmap(srcMat, result, true)
    srcMat.release()
    return result
}

private fun findContours(src: Mat): ArrayList<MatOfPoint> {

    val grayImage: Mat
    val threshImage : Mat

    val cannedImage: Mat
    val kernel: Mat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
    val dilate: Mat
    val size = Size(src.size().width, src.size().height)

    grayImage = Mat(size, CvType.CV_8UC1)
    threshImage = Mat(size, CvType.CV_8UC1)
    cannedImage = Mat(size, CvType.CV_8UC1)
    dilate = Mat(size, CvType.CV_8UC1)

    Imgproc.cvtColor(src, grayImage, Imgproc.COLOR_BGR2GRAY)
    // Imgproc.GaussianBlur(grayImage, grayImage, Size(5.0, 5.0), 0.0)
    Imgproc.threshold(grayImage, threshImage, 150.0, 255.0, Imgproc.THRESH_BINARY)
    // Imgproc.Canny(grayImage, cannedImage, 75.0, 200.0)
    // Imgproc.dilate(cannedImage, dilate, kernel)
    val contours = ArrayList<MatOfPoint>()
    val hierarchy = Mat()

    Imgproc.findContours(
            threshImage,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_NONE
    )
    contours.sortByDescending { p: MatOfPoint -> Imgproc.contourArea(p) }
    hierarchy.release()
    grayImage.release()
    threshImage.release()
    // cannedImage.release()
    // kernel.release()
    // dilate.release()


    return contours
}

private fun getCorners(contours: ArrayList<MatOfPoint>, size: Size): Corners? {
    
// int width = this.getResources().getDisplayMetrics().widthPixels;
    
    
    val indexTo: Int = when (contours.size) {
        in 0..5 -> contours.size - 1
        else -> 4
    }

    var max_area = 0.0
    var points : List<Point> = listOf()
    var pointsToDraw : List<Point> = listOf()
    val approx = MatOfPoint2f()

    for (contour in contours){
        val c2f = MatOfPoint2f(*contour.toArray())
        val area = Imgproc.contourArea(c2f)
        if (area > 100){
            val peri = Imgproc.arcLength(c2f,true)
            Imgproc.approxPolyDP(c2f,approx,0.1*peri,true)
            points = approx.toArray().asList()
            val rect = Imgproc.boundingRect(contour)
           
            if (area > max_area && points.size==4 && rect.width > size.width*0.3 && (((rect.x + rect.width/2)-(size.width/2))> -100) && (((rect.x + rect.width/2)-(size.width/2))<100)
             && (((rect.y + rect.height/2)-(size.height/2))> -300) && (((rect.y + rect.height/2)-(size.height/2))< 300) ){
                 println("worked")
                max_area = area
                pointsToDraw=points
                }
                
        }
    }
    
  
     

    if(pointsToDraw.size==0){
        return null
    }
    else{
        return Corners(pointsToDraw,size)
       
        
    

    }











//=========------===========

// var maxAreInHistory = 0.0
//      if(pointsToDraw.size==0){
// return null
//      } 

//      else{

// for(area in areaHistory){

//       if(area>maxAreInHistory){
//           maxAreInHistory=area
//       }

// }

//    if(max_area>=maxAreInHistory){
// return Corners(pointsToDraw, size)
//    }
//    else {
// return null
//    }
}





private fun sortPoints(points: List<Point>): List<Point> {
    val p0 = points.minByOrNull { point -> point.x + point.y } ?: Point()
    val p1 = points.minByOrNull { point: Point -> point.y - point.x } ?: Point()
    val p2 = points.maxByOrNull { point: Point -> point.x + point.y } ?: Point()
    val p3 = points.maxByOrNull { point: Point -> point.y - point.x } ?: Point()
    return listOf(p0, p1, p2, p3)
}
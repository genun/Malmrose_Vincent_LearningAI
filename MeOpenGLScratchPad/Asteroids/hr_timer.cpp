#include <windows.h>
#include "hr_timer.h"
 
 double CStopWatch::LIToSecs( LARGE_INTEGER & L) {
     return ((double)L.QuadPart /(double)frequency.QuadPart) ;
 }
 
 CStopWatch::CStopWatch(){
     timer.start.QuadPart=0;
     timer.stop.QuadPart=0; 
     QueryPerformanceFrequency( &frequency );
	 QueryPerformanceCounter(&inter);
 }
 
 void CStopWatch::startTimer( ) {
     QueryPerformanceCounter(&timer.start) ;
 }
 
 void CStopWatch::stopTimer( ) {
     QueryPerformanceCounter(&timer.stop) ;
 }
 
 double CStopWatch::getElapsedTime() {
     LARGE_INTEGER time;
     time.QuadPart = timer.stop.QuadPart - timer.start.QuadPart;
     return LIToSecs( time) ;
 }

double CStopWatch::interval(){
	QueryPerformanceCounter(&temp);
	LARGE_INTEGER temp2;
	temp2.QuadPart = temp.QuadPart - inter.QuadPart;
	inter = temp;
	return LIToSecs(temp2);
}

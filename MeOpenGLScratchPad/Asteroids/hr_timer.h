 #ifndef hr_timer
 #define hr_timer
 #include <windows.h>
 
 typedef struct {
     LARGE_INTEGER start;
     LARGE_INTEGER stop;
 } stopWatch;
 
 struct CStopWatch {
    stopWatch timer;
    LARGE_INTEGER frequency;
	LARGE_INTEGER inter;
	LARGE_INTEGER temp;
    double LIToSecs( LARGE_INTEGER & L) ;
    CStopWatch() ;
    void startTimer( ) ;
    void stopTimer( ) ;
    double getElapsedTime() ;
	double interval();
 };

 #endif
 
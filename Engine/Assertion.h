#ifndef ASSERTION_H
#define ASSERTION_H
#include "Logger.h"
#include "Engine.h"


	#ifdef ASSERTION_ON
		#define ASSERT(expr, ...)  {const char* msg = #expr##" "##__VA_ARGS__; if (!(expr)) {LOG(Severe, msg ); END_LOG DebugBreak(); exit(1);}}
	#else
		#pragma warning(disable: 4127)
		#define ASSERT (expr, ...) do{}while(0);
		#pragma warning(default: 4127)
	#endif 

#endif

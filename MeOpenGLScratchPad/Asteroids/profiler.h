#ifndef PROFILER_H
#define PROFILER_H
#include <fstream>



struct profiler
{
#if PROFILE_ON
	unsigned int input;
	static const unsigned int possibleinput = 15000;
	double numInput[possibleinput];
	void initialize();
	void shutdown();
	void addEntry(double time);
#else
	unsigned int input;
	static const unsigned int possibleinput = 15000;
	double numInput[possibleinput];
	void initialize(){}
	void shutdown(){}
	void addEntry(double time){time;}
#endif
};

#endif
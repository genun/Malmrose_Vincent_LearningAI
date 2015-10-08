#if PROFILE_ON
#include "profiler.h"

void profiler::addEntry(double time){
	if(input < possibleinput){
		numInput[input] = time;
		input ++;
	}
}	
	
void profiler::initialize()
{
	input = 0;
}

void profiler::shutdown()
{
	std::ofstream myfile;
	myfile.open("CurrentProfileData.csv");
	myfile << "Update Ship, Update Input, Update LERPER, Update Turret, Update Bullet, Update flames, Update orbit, Update effects,";
	myfile << "Draw Ship, Draw Orbit, Draw LERPER, Draw Turret, Draw Bullet, Draw boarder, Draw effects, Draw text, Draw FPS\n";
	for (unsigned int i = 0; i < input; i ++){
		if(i % 17 == 0 && i != 0) {
			myfile << numInput[i] << "\n";
		}
		else{
			myfile << numInput[i] << ",";
		}
	}
	myfile.close();
	int g;
	g = 0;
}

#endif
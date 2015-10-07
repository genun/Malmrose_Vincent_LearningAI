#ifndef SLIDER_INFO
#define SLIDER_INFO
#include <DebugTools\meSlider.h>

class SliderInfo{
	DebugSlider* slider;
	float* value;
	friend class DebugMenu;
public:
	void init(float theMin, float theMax, float* theValue);
	void updateFloat();
};

#endif

#include <DebugTools\Menu\Infos\SliderInfo.h>

void SliderInfo::init(float theMin, float theMax, float* theValue){
	slider = new DebugSlider(theMin, theMax, false);
	value = theValue;
	slider->setValue(*value);
}

void SliderInfo::updateFloat(){
	*value = slider->value();
}

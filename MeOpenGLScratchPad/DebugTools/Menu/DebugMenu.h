#ifndef DEBUG_TOOLS_MENU
#define DEBUG_TOOLS_MENU
#include <GL/glew.h>
#include <DebugTools\Menu\Infos\WatchInfo.h>
#include <DebugTools\Menu\Infos\CheckBoxInfo.h>
#include <DebugTools\Menu\Infos\SliderInfo.h>
#include <Qt\qlist.h>
#include <QtOpenGL\qglwidget>
class QVBoxLayout;
 
class DebugMenu: QGLWidget{

	QVBoxLayout* layout;
	QList<WatchInfo> watchInfos;
	QList<SliderInfo> sliderInfos;
	QList<CheckBoxInfo> checkInfos;
	float time;

	static DebugMenu* instance;

	DebugMenu(){}
	DebugMenu(const DebugMenu&);
	DebugMenu& operator=(const DebugMenu&);
public:
	static DebugMenu& getInstance(){ return *instance;}
	void watch(const char* text, const float& value);
	void slideFloat(const char* text, float* value, 
						   float min, float max);
	void DebugMenu::checkBox(const char* text, bool* value);
	bool initialize(QVBoxLayout* theLayout);
	void update();

};

#define debugMenu DebugMenu::getInstance()

#endif 
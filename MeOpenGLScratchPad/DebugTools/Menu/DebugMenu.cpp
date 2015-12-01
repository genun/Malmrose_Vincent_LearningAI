#include <DebugTools\Menu\DebugMenu.h>
#include <Qt\qlabel.h>
#include <QtGui\qvboxlayout>
#include <QtGui\qhboxlayout>

DebugMenu* DebugMenu::instance;

void DebugMenu::watch(const char* text, const float& value){
	QHBoxLayout* row;
	layout->addLayout(row = new QHBoxLayout);
	row->addWidget(new QLabel(text));
	WatchInfo w;
	w.myFloat = &value;
	row->addWidget(w.myLabel = new QLabel());
	watchInfos.append(w);
}

void DebugMenu::update(){
	for(int i = 0; i < watchInfos.size(); i++){
		WatchInfo& w = watchInfos[i];
		w.myLabel->setText(QString::number(*(w.myFloat)));
	}
	for(int i = 0; i < sliderInfos.size(); i++){
		SliderInfo& s = sliderInfos[i];
		s.updateFloat();
	}
	for(int i = 0; i < checkInfos.size(); i++){
		CheckBoxInfo c = checkInfos[i];
		c.updateCheckBox();
	}

	if(GetAsyncKeyState(VK_TAB) && time < 0 ){
		//if(layout->parentWidget()->isHidden()){
		//	layout->parentWidget()->show();
		//	time = 1;
		//}
		//else{
		//	layout->parentWidget()->hide();
		//	time = 1;
		//}
	}
	
	if(time > 0)
		time -= 0.1f;
}

void DebugMenu::slideFloat(const char* text, float* value, float min, float max){
	QHBoxLayout* row;
	layout->addLayout(row = new QHBoxLayout);
	row->addWidget(new QLabel(text));

	SliderInfo s;
	s.init(min, max, value);
	row->addWidget(s.slider);
	sliderInfos.append(s);
}

bool DebugMenu::initialize(QVBoxLayout* theLayout){
	/*assert(theLayout != 0);*/

	if (instance != 0)
		return false;
	instance = new DebugMenu();
	instance->layout = theLayout;
	instance->time = 1.0f;
}

void DebugMenu::checkBox(const char* text, bool* value){
	QHBoxLayout* row;
	layout->addLayout(row = new QHBoxLayout);
	row->addWidget(new QLabel(text));
	CheckBoxInfo c;
	c.init(value);
	row->addWidget(c.checkBox);
	checkInfos.append(c);
}

bool DebugMenu::shutdown(){
	if (instance == 0)
		return false;
	delete instance;
	instance = 0;
	return true;
}

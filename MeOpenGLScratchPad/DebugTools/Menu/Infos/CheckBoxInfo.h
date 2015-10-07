#ifndef CHECK_BOX_INFO
#define CHECK_BOX_INFO
#include <Qt\qcheckbox.h>

class CheckBoxInfo{
	QCheckBox* checkBox;
	bool* value;
	friend class DebugMenu;
public:
	void init(bool* value);
	void updateCheckBox();
};

#endif
#include <DebugTools\Menu\Infos\CheckBoxInfo.h>

void CheckBoxInfo::init(bool* theValue){
	value = theValue;
	checkBox = new QCheckBox();
	if(theValue)
		checkBox->setCheckState(Qt::Checked);
	else
		checkBox->setCheckState(Qt::Unchecked);
}

void CheckBoxInfo::updateCheckBox(){
	if(checkBox->checkState() == Qt::Checked)
		*value = true;
	else
		*value = false;
}

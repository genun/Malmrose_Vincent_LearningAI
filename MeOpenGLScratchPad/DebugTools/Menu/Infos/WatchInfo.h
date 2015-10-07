#ifndef DEBUG_TOOLS_WATCH_INFO
#define DEBUG_TOOLS_WATCH_INFO
class QLabel;

class WatchInfo{

	const float* myFloat;
	QLabel* myLabel;

	friend class DebugMenu;
public:
};

#endif